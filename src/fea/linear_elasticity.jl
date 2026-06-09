# task local data for parallel stiffness assemble
struct StiffnessScratch{CC<:CellCache,CV<:CellValues,JCFG<:ForwardDiff.JacobianConfig,A<:Ferrite.AbstractAssembler}
    cell_cache::CC
    cellvalues::CV
    Ke::Matrix{Float64}
    jac::Matrix{Float64}
    cfg::JCFG
    assembler::A
end
function StiffnessScratch(model::FEModel{dim,nvar}, asm::Ferrite.AbstractAssembler) where {dim,nvar}
    cell_cache = CellCache(model.dh)
    cellvalues = copy(model.cellvalues)
    n_basefuncs = getnbasefunctions(cellvalues)

    Ke = zeros(n_basefuncs, n_basefuncs)
    jac = zeros(n_basefuncs * n_basefuncs, nvar)
    cfg = ForwardDiff.JacobianConfig(
        (Ke, xe) -> element_stiffness!(Ke, xe, cell_cache, model),
        Ke, zeros(nvar),
        ForwardDiff.Chunk{nvar}()
    )

    return StiffnessScratch(cell_cache, cellvalues, Ke, jac, cfg, asm)
end

"""
Contains the data necessary for the linear system solve and stores the results
"""
struct LinearElasticity{FEM<:FEModel,SS<:StiffnessScratch}
    model::FEM
    ps::MKLPardisoSolver

    x::Vector{Float64}
    K::SparseMatrixCSC{Float64,Int}
    ∂Ke∂x::Vector{Matrix{Float64}}
    solution::Vector{Float64}
    chnl::Channel{SS}
end

"""
    LinearElasticity(model::FEModel)

Creates a new `LinearElasticity` with uninitialized design variables
"""
function LinearElasticity(model::FEModel)
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    set_iparm!(ps, 1, 1) # to be able to manually set iparm
    set_iparm!(ps, 12, 1) # tells Pardiso we are giving a CSC matrix instead of CSR

    # preallocate vectors
    x0 = Vector{Float64}(undef, getncells(model.grid) * get_nvar(model))
    solution = similar(x0, ndofs(model.dh))

    # preallocate stiffness and sensitivities
    n_basefuncs = getnbasefunctions(model.cellvalues)
    K = allocate_matrix(model.dh)
    ∂Ke∂x = fill(zeros(n_basefuncs, n_basefuncs), length(x0))

    # preallocate thread local containers
    chnl = Channel{StiffnessScratch}(Threads.nthreads())
    foreach(1:Threads.nthreads()) do _
        asm = start_assemble(K; fillzero=false)
        put!(chnl, StiffnessScratch(model, asm))
    end

    return LinearElasticity(model, ps, x0, K, ∂Ke∂x, solution, chnl)
end

"""
    solve!(solver::LinearElasticity, f::AbstractVector)

Solves the finite element problem with the stiffness matrix stored in `solver` and right hand side `f`.
Stores the result in `solver.solution`
"""
function solve!(solver::LinearElasticity, f::AbstractVector)
    @unpack K, solution, ps = solver

    # solve linear system and store result in solver.solution
    pardiso(ps, solution, tril(K), f)
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE) # reuse K for next solves
end

"""
    adjoint_sensitivities!(dJdx, lambda, displacements, solver::FESolver)

Uses the adjoint system to compute the sensitivities ∂J/∂x of a function J(u).

`lambda` is the vector of adjoint variables, obtained from calling `fea!` with right hand side dJ/du.
`displacements` is the vector of nodal displacements, obtained from calling `fea!` with right hand side equal to the forces vector
"""
function adjoint_sensitivities!(dJdx, lambda, displacements, solver::LinearElasticity)
    model = solver.model
    ∂Ke∂x = solver.∂Ke∂x
    for cell in CellIterator(model.dh)
        e = cellid(cell)
        dofs = celldofs(cell)

        @views ue, lambdae = lambda[dofs], displacements[dofs]
        nvar = get_nvar(model)
        for i in 1:nvar
            dJdx[nvar*(e-1)+i] = -dot(lambdae, ∂Ke∂x[nvar*(e-1)+i], ue)
        end
    end
end

"""
    update_stiffness!(solver::LinearElasticity, x::AbstractVector)

Recomputes the stiffness matrix, using the new design variables `x`.
Needed to update the stiffness matrix before calling `fea!` and `adjoint_sensitivities!`
"""
function update_stiffness!(solver::LinearElasticity, x::AbstractVector)
    solver.x .= x

    @unpack ps, x, K, ∂Ke∂x, chnl = solver
    model = solver.model

    n_basefuncs = getnbasefunctions(model.cellvalues)
    nvar = get_nvar(model)

    start_assemble(K) # zero K out before starting
    for color in model.colors
        @tasks for e in color
            scratch = take!(chnl)
            @unpack cell_cache, cellvalues, Ke, jac, cfg, assembler = scratch

            Ferrite.reinit!(cell_cache, e)
            Ferrite.reinit!(cellvalues, cell_cache)

            xe = @view x[nvar*(e-1)+1:nvar*e]

            # update Ke and obtain ∂Ke∂x via automatic differentiation
            ForwardDiff.jacobian!(
                jac,
                (Ke, xe) -> element_stiffness!(Ke, xe, cellvalues, model),
                Ke, xe, cfg,
                Val{false}() # disable tag checking because we are rebuilding the anonymous function
            )
            for var_idx = 1:nvar
                @views ∂Ke∂x[nvar*(e-1)+var_idx] = reshape(jac[:, var_idx], n_basefuncs, n_basefuncs)
            end

            assemble!(assembler, celldofs(cell_cache), Ke)
            put!(chnl, scratch)
        end
    end

    # apply boundary conditions
    apply!(K, model.ch)

    # tell Pardiso to recalculate the numerical factorization in the next solve
    # the sparsity pattern of K doesnt change, skip the symbolic factorization
    if get_phase(ps) == Pardiso.SOLVE_ITERATIVE_REFINE
        set_phase!(ps, Pardiso.NUM_FACT_SOLVE_REFINE)
    end
end

# xe is a view of the design vector that contains the variables corresponding to one element
function element_stiffness!(Ke::Matrix{T}, xe::AbstractVector, cellvalues::CellValues, model::FEModel) where {T<:Real}
    fill!(Ke, zero(T))

    @inbounds for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            ∇sδεi = shape_symmetric_gradient(cellvalues, q_point, i)
            for j in 1:i
                ∇sδεj = shape_symmetric_gradient(cellvalues, q_point, j)
                Ke[i, j] += ∇sδεi ⊡ interpolate(xe, model.mat_interp) ⊡ ∇sδεj * dΩ
            end
        end
    end

    for i in axes(Ke, 1), j in axes(Ke, 1)[begin+i:end]
        Ke[i, j] = Ke[j, i]
    end
end
