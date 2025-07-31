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

Need to be mutable to allow a finalizer to be set. It will tell Pardiso to free all memory when the optimization is done
"""
mutable struct FESolver{T<:Real,VT<:AbstractVector{T},FEM<:FEModel,SS<:MKLPardisoSolver,MT<:SparseMatrixCSC{T},SD<:ScratchData}
    model::FEM
    ps::SS
    xPhys::VT

    K::MT
    f::Vector{T}
    chnl::Channel{SD}

    ∂Ke∂x::Vector{Matrix{T}}
    u::Vector{T}
end

"""
    FESolver(x0::AbstractVector, model::FEModel)

Creates a new `FESolver` with the design variables initialized at `x0`

# NOTE: resets `TopOpt.timer`
"""
function FESolver(x0::AbstractVector, model::FEModel)
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    set_iparm!(ps, 1, 1) # to be able to manually set iparm
    set_iparm!(ps, 12, 1) # tells Pardiso we are giving a CSC matrix instead of CSR

    reset_timer!(TopOpt.timer)

    # compute RHS, only done once assuming loads do not depend on the design
    f = compute_force_vector(model)
    apply!(f, model.ch)

    # preallocate stiffness and sensitivities
    n_basefuncs = getnbasefunctions(model.cellvalues)
    K = allocate_matrix(model.dh)
    ∂Ke∂x = fill(zeros(n_basefuncs, n_basefuncs), length(x0))

    # preallocate solution
    u = zeros(ndofs(model.dh))

    # preallocate thread local containers
    chnl = Channel{ScratchData}(Threads.nthreads())
    foreach(1:Threads.nthreads()) do _
        put!(chnl, ScratchData(model, K))
    end

    solver = FESolver(model, ps, copy(x0), K, f, chnl, ∂Ke∂x, u)
    finalizer(solver) do s
        set_phase!(s.ps, Pardiso.RELEASE_ALL)
    end
end

"""
    fea!(solver::FESolver[, x::AbstractVector])

Solves the finite element problem with variables `x` and overwrites the results to the memory preallocated in `solver`

If `x` is not given, it uses the same values already stored in `solver`
If `x` is not a `DesignVector`, it assumes that each finite element is represented by a single design variable
"""
function fea!(solver::FESolver, x::AbstractVector)
    solver.xPhys .= x
    fea!(solver)
end
function fea!(solver::FESolver)
    @unpack K, f, u, ps = solver

    @timeit timer "stiffness assemble" begin
        global_stiffness!(solver)
        apply!(K, solver.model.ch)
    end

    @timeit timer "linear solve" begin
        # solve linear system and store result in solver.u
        pardiso(ps, u, tril(K), f)

        # the sparsity pattern of K doesnt change, so next solves can skip the symbolic factorization step
        set_phase!(ps, Pardiso.NUM_FACT_SOLVE_REFINE)
    end
end

function global_stiffness!(solver::FESolver)
    @unpack xPhys, K, ∂Ke∂x, chnl = solver
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
            xe = element_slice(xPhys, e)

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

function compute_force_vector(model::FEModel{dim}) where {dim}
    f = zeros(ndofs(model.dh))
    fe = zeros(getnbasefunctions(model.facetvalues))

    for cell in CellIterator(model.dh)
        dofs = celldofs(cell)

        # nodal forces
        for force in model.loads
            force isa NodalLoad || continue

            for (i, node) in enumerate(getnodes(cell))
                node in getnodeset(model.grid, force.nodeset_name) || continue
                f[dofs[dim*(i-1)+1:dim*i]] .+= force.F
            end
        end

        # linear forces
        for force in model.loads
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

    return f
end
