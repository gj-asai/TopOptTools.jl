# task local data for parallel stiffness assemble
# IMPORTANT: each task works with its own copy of cellvalues and creates a local CellCache
struct ScratchData{CC,CV,T,A}
    cell_cache::CC
    cellvalues::CV
    Ke::Matrix{T}
    jac::Matrix{T}
    assembler::A
end

function ScratchData(model::FEModel{dim,nvar}, K::SparseMatrixCSC) where {dim,nvar}
    cell_cache = CellCache(model.dh)
    cellvalues = copy(model.cellvalues)

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    jac = zeros(n_basefuncs * n_basefuncs, nvar)

    asm = start_assemble(K; fillzero=false)
    return ScratchData(cell_cache, cellvalues, Ke, jac, asm)
end

"""
Contains the data necessary for the linear system solve and stores the results
"""
struct FESolver{T<:Real,VT<:AbstractVector{T},FEM<:FEModel,SS<:MKLPardisoSolver,MT<:SparseMatrixCSC{T},SD<:ScratchData}
    model::FEM
    ps::SS

    x::VT
    K::MT
    ∂Ke∂x::Vector{Matrix{T}}
    solution::Vector{T}
    chnl::Channel{SD}
end

"""
    FESolver(x0::AbstractVector, model::FEModel)

Creates a new `FESolver` with the design variables initialized at `x0`
"""
function FESolver(x0::AbstractVector, model::FEModel)
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    set_iparm!(ps, 1, 1) # to be able to manually set iparm
    set_iparm!(ps, 12, 1) # tells Pardiso we are giving a CSC matrix instead of CSR

    # preallocate stiffness and sensitivities
    n_basefuncs = getnbasefunctions(model.cellvalues)
    K = allocate_matrix(model.dh)
    ∂Ke∂x = fill(zeros(n_basefuncs, n_basefuncs), length(x0))

    # preallocate solution
    solution = zeros(ndofs(model.dh))

    # preallocate thread local containers
    chnl = Channel{ScratchData}(Threads.nthreads())
    foreach(1:Threads.nthreads()) do _
        put!(chnl, ScratchData(model, K))
    end

    return FESolver(model, ps, copy(x0), K, ∂Ke∂x, solution, chnl)
end

"""
    fea!(solver::FESolver, f::AbstractVector)

Solves the finite element problem with the stiffness matrix stored in `solver` and right hand side `f`.
Stores the result in `solver.solution`
"""
function fea!(solver::FESolver, f::AbstractVector)
    @unpack K, solution, ps = solver

    # solve linear system and store result in solver.u
    pardiso(ps, solution, tril(K), f)

    # the sparsity pattern of K doesnt change, so next solves can skip the symbolic factorization step
    # this is also the best option if K is exactly the same as in the previous solve
    set_phase!(ps, Pardiso.NUM_FACT_SOLVE_REFINE)
end

"""
    adjoint_sensitivities!(dfdx, lambda, displacements, solver::FESolver)

Uses the adjoint system to compute the sensitivities ∂f/∂x of a function f(u).

`lambda` is the vector of adjoint variables, obtained from calling `fea!` with right hand side df/du.
`displacements` is the vector of nodal displacements, obtained from calling `fea!` with right hand side equal to the forces vector
"""
function adjoint_sensitivities!(dfdx, lambda, displacements, solver::FESolver)
    model = solver.model
    ∂Ke∂x = solver.∂Ke∂x
    for cell in CellIterator(model.dh)
        e = cellid(cell)
        dofs = celldofs(cell)

        @views ue, lambdae = lambda[dofs], displacements[dofs]
        nvar = get_nvar(model)
        for i in 1:nvar
            dfdx[nvar*(e-1)+i] = -dot(lambdae, ∂Ke∂x[nvar*(e-1)+i], ue)
        end
    end
end

"""
    update_stiffness!(solver::FESolver, x::AbstractVector)

Recomputes the stiffness matrix, using the new design variables `x`.

This function must always be called before using `fea!` and `adjoint_sensitivities!`, otherwise the system will be solved using the wrong stiffness matrix
"""
function update_stiffness!(solver::FESolver, x::AbstractVector)
    solver.x .= x

    @unpack x, K, ∂Ke∂x, chnl = solver
    model = solver.model

    n_basefuncs = getnbasefunctions(model.cellvalues)
    nvar = get_nvar(model)

    start_assemble(K) # zero K out before starting
    for color in model.colors
        @tasks for e in color
            scratch = take!(chnl)
            @unpack cell_cache, cellvalues, Ke, jac, assembler = scratch

            Ferrite.reinit!(cell_cache, e)
            Ferrite.reinit!(cellvalues, cell_cache)

            xe = @view x[nvar*(e-1)+1:nvar*e]

            # this updates Ke and obtains ∂Ke∂x via automatic differentiation
            # uses forward mode because the number of outputs is in general larger than the number of inputs:
            # Inputs: nvar
            # Outputs: entries of Ke, e.g. 64 for 2D linear quadrilateral elements
            ForwardDiff.jacobian!(jac, (Ke, xe) -> element_stiffness!(Ke, xe, cellvalues, model), Ke, xe)
            for var_idx = 1:nvar
                ∂Ke∂x[nvar*(e-1)+var_idx] = reshape(jac[:, var_idx], n_basefuncs, n_basefuncs)
            end

            assemble!(assembler, celldofs(cell_cache), Ke)
            put!(chnl, scratch)
        end
    end

    # apply boundary conditions
    apply!(K, model.ch)
end

# xe is a view of the design vector that contains the variables corresponding to one element
function element_stiffness!(Ke::Matrix{T}, xe::AbstractVector, cellvalues::CellValues, model::FEModel) where {T<:Real}
    fill!(Ke, zero(T))

    @inbounds for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            δεi = shape_symmetric_gradient(cellvalues, q_point, i)
            for j in 1:i
                δεj = shape_symmetric_gradient(cellvalues, q_point, j)
                Ke[i, j] += δεi ⊡ interpolate(xe, model.mat_interp) ⊡ δεj * dΩ
            end
        end
    end

    for i in 1:size(Ke, 1), j in i+1:size(Ke, 1)
        Ke[i, j] = Ke[j, i]
    end
end

"""
    compute_force_vector(loads::Vector{<:Load}, model::FEModel)

Returns the nodal forces vector corresponding to the load case `loads`, with the Dirichlet boundary conditions in `model` already applied
"""
function compute_force_vector(loads::Vector{<:Load}, model::FEModel{dim}) where {dim}
    f = zeros(ndofs(model.dh))
    fe = zeros(getnbasefunctions(model.facetvalues))

    for cell in CellIterator(model.dh)
        dofs = celldofs(cell)

        # nodal forces
        for force in loads
            force isa NodalLoad || continue

            for (i, node) in enumerate(getnodes(cell))
                node in getnodeset(model.grid, force.nodeset_name) || continue
                f[dofs[dim*(i-1)+1:dim*i]] .+= force.F
            end
        end

        # linear forces
        for force in loads
            force isa LinearLoad || continue

            fill!(fe, 0.0)
            for facet in 1:nfacets(cell)
                (cellid(cell), facet) in getfacetset(model.grid, force.faceset_name) || continue

                Ferrite.reinit!(model.facetvalues, cell, facet)
                for q_point in 1:getnquadpoints(model.facetvalues)
                    dΓ = getdetJdV(model.facetvalues, q_point)
                    for i in 1:getnbasefunctions(model.facetvalues)
                        δu = shape_value(model.facetvalues, q_point, i)
                        fe[i] += (δu ⋅ force.F) * dΓ
                    end
                end
            end
            assemble!(f, dofs, fe)
        end
    end

    apply!(f, model.ch)
    return f
end
