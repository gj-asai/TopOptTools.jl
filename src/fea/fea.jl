"""
Contains the task local data for parallel stiffness assemble
"""
struct StiffnessScratch{CC<:CellCache,CV<:CellValues,JCFG<:ForwardDiff.JacobianConfig,A<:Ferrite.AbstractAssembler}
    cell_cache::CC
    cellvalues::CV
    Ke::Matrix{Float64}
    jac::Matrix{Float64}
    cfg::JCFG
    assembler::A
end
function StiffnessScratch(model::FEModel, asm::Ferrite.AbstractAssembler, nvar::Int)
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

abstract type StiffnessBuilder{nvar} end

"""
Contains the data necessary for the linear system solve and stores the results.
`stiff_builder` contains the physics that will be used to assemble the stiffness matrix.
"""
struct FEA{nvar,FEM<:FEModel,SB<:StiffnessBuilder{nvar},SS<:StiffnessScratch}
    model::FEM
    ps::MKLPardisoSolver

    x::Vector{Float64}
    stiff_builder::SB

    K::SparseMatrixCSC{Float64,Int}
    ∂Ke∂x::Vector{Matrix{Float64}}
    solution::Vector{Float64}
    chnl::Channel{SS}
end
function FEA(model::FEModel, stiff_builder::StiffnessBuilder{nvar}) where {nvar}
    # preallocate vectors
    x0 = Vector{Float64}(undef, getncells(model.grid) * nvar)
    solution = similar(x0, ndofs(model.dh))

    # preallocate stiffness and sensitivities
    n_basefuncs = getnbasefunctions(model.cellvalues)
    K = allocate_matrix(model.dh)
    ∂Ke∂x = fill(zeros(n_basefuncs, n_basefuncs), length(x0))

    # preallocate thread local containers
    chnl = Channel{StiffnessScratch}(Threads.nthreads())
    foreach(1:Threads.nthreads()) do _
        asm = start_assemble(K; fillzero=false)
        put!(chnl, StiffnessScratch(model, asm, nvar))
    end

    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    set_iparm!(ps, 1, 1) # to be able to manually set iparm
    set_iparm!(ps, 12, 1) # tells Pardiso we are giving a CSC matrix instead of CSR

    return FEA(model, ps, x0, stiff_builder, K, ∂Ke∂x, solution, chnl)
end

"""
    solve!(fea::FEA, f::AbstractVector)

Solves the finite element problem with the stiffness matrix stored in `fea` and right hand side `f`.
Stores the result in `fea.solution`
"""
function solve!(fea::FEA, f::AbstractVector)
    pardiso(fea.ps, fea.solution, tril(fea.K), f)
    set_phase!(fea.ps, Pardiso.SOLVE_ITERATIVE_REFINE) # reuse K for next solves
end

"""
    adjoint_sensitivities!(dJdx, lambda, displacements, fea::FEA)

Uses the adjoint system to compute the sensitivities ∂J/∂x of a function J(u).

`lambda` is the vector of adjoint variables, obtained from calling `solve!` with right hand side dJ/du.
`displacements` is the vector of nodal displacements, obtained from calling `solve!` with right hand side equal to the forces vector
"""
function adjoint_sensitivities!(dJdx, lambda, displacements, fea::FEA{nvar}) where {nvar}
    model = fea.model
    ∂Ke∂x = fea.∂Ke∂x
    for cell in CellIterator(model.dh)
        e = cellid(cell)
        dofs = celldofs(cell)

        @views ue, lambdae = displacements[dofs], lambda[dofs]
        for i in 1:nvar
            dJdx[nvar*(e-1)+i] = -dot(lambdae, ∂Ke∂x[nvar*(e-1)+i], ue)
        end
    end
end

"""
    update_stiffness!(fea::FEA, x::AbstractVector)

Recomputes the stiffness matrix, using the new design variables `x`.
Needed to update the stiffness matrix before calling `solve!` and `adjoint_sensitivities!`
"""
function update_stiffness!(fea::FEA{nvar}, x::AbstractVector) where {nvar}
    fea.x .= x

    @unpack x, K, ∂Ke∂x, chnl = fea
    model = fea.model

    n_basefuncs = getnbasefunctions(model.cellvalues)

    start_assemble(K) # zero K out before starting
    for color in model.colors
        Threads.@threads for e in color
            scratch = take!(chnl)
            @unpack cell_cache, cellvalues, Ke, jac, cfg, assembler = scratch

            Ferrite.reinit!(cell_cache, e)
            Ferrite.reinit!(cellvalues, cell_cache)

            xe = @view x[nvar*(e-1)+1:nvar*e]

            # update Ke and obtain ∂Ke∂x via automatic differentiation
            ForwardDiff.jacobian!(
                jac,
                (Ke, xe) -> element_stiffness!(Ke, xe, cellvalues, fea.stiff_builder),
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
    if get_phase(fea.ps) == Pardiso.SOLVE_ITERATIVE_REFINE
        set_phase!(fea.ps, Pardiso.NUM_FACT_SOLVE_REFINE)
    end
end

# xe is a view of the design vector that contains the variables corresponding to one element
function element_stiffness!(::Matrix, ::AbstractVector, ::CellValues, stiff_builder::StiffnessBuilder)
    throw("Method element_stiffness!(Ke::Matrix, xe::AbstractVector, cv::CellValues, ::$(typeof(stiff_builder))) is not defined")
end
