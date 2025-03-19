struct FEResults{T}
    K::SparseMatrixCSC{T}
    f::Vector{T}
    ∂Ke∂x::Vector{Matrix{T}}
    u::Vector{T}
end

function FEResults(model::FEModel)
    f = compute_force_vector(model)
    apply!(f, model.ch)

    # preallocate stiffness and sensitivities
    n_basefuncs = getnbasefunctions(model.cellvalues)
    K = allocate_matrix(model.dh)
    ∂Ke∂x = fill(zeros(n_basefuncs, n_basefuncs), get_nvar(model) * getncells(model.grid))

    # preallocate solution
    u = zeros(get_dim(model) * getnnodes(model.grid))

    return FEResults(K, f, ∂Ke∂x, u)
end

# task local data for parallel stiffness assemble
struct ScratchData{CC,CV,T,A}
    cell_cache::CC
    cellvalues::CV
    Ke::Matrix{T}
    assembler::A
end

function ScratchData(model::FEModel, K::SparseMatrixCSC)
    cell_cache = CellCache(model.dh)
    n_basefuncs = getnbasefunctions(model.cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    asm = start_assemble(K; fillzero=false)
    return ScratchData(cell_cache, copy(model.cellvalues), Ke, asm)
end

function fea!(results::FEResults, x::Vector, model::FEModel)
    @unpack K, f, ∂Ke∂x, u = results

    # assemble linear system, uses multithreading
    global_stiffness!(K, ∂Ke∂x, x, model)
    apply!(K, model.ch)

    # solve linear system, conjugate gradients from Krylov.jl
    # avoiding u .= K\f because it uses OpenBLAS which conflicts with the multithreading when running in the cluster
    u .= cg(K, f, u)[1]
end

function global_stiffness!(K, ∂Ke∂x, x::Vector, model::FEModel)
    n_basefuncs = getnbasefunctions(model.cellvalues)
    nvar = get_nvar(model)

    start_assemble(K) # zero K out
    for color in model.colors
        @tasks for cellidx in color
            @set scheduler = :static # stick tasks to threads
            @local scratch = ScratchData(model, K)
            @unpack cell_cache, cellvalues, Ke, assembler = scratch

            reinit!(cell_cache, cellidx)
            reinit!(cellvalues, cell_cache)

            e = cellidx
            xe = x[nvar*(e-1)+1:nvar*e]

            jac = ForwardDiff.jacobian((Ke, xe) -> element_stiffness!(Ke, xe, cellvalues, model), Ke, xe)
            for var_idx = 1:nvar
                @views ∂Ke∂x[nvar*(e-1)+var_idx] = reshape(jac[:, var_idx], (n_basefuncs, n_basefuncs))
            end

            assemble!(assembler, celldofs(cell_cache), Ke)
        end
    end
end

function element_stiffness!(Ke::Matrix{T}, xe::AbstractVector{T}, cellvalues::CellValues, model::FEModel) where {T<:Real}
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

    Ke .= Symmetric(Ke, :L)
end

function compute_force_vector(model::FEModel)
    f = zeros(ndofs(model.dh))
    dim = get_dim(model)

    # nodal forces
    for cell in CellIterator(model.dh)
        dofs = celldofs(cell)
        for (i, node) in enumerate(getnodes(cell)), force in model.loads
            force isa NodalLoad || continue
            node in getnodeset(model.grid, force.nodeset_name) || continue
            f[dofs[dim*(i-1)+1:dim*i]] .+= force.F
        end
    end

    # linear forces
    fe = zeros(getnbasefunctions(model.facetvalues))
    for force in model.loads
        force isa LinearLoad || continue
        for facet in FacetIterator(model.dh, force.faceset_name)
            reinit!(model.facetvalues, facet)
            fill!(fe, 0.0)
            for q_point in 1:getnquadpoints(model.facetvalues)
                dΓ = getdetJdV(model.facetvalues, q_point)
                for i in 1:getnbasefunctions(model.facetvalues)
                    δu = shape_value(model.facetvalues, q_point, i)
                    fe[i] += (δu ⋅ force.F) * dΓ
                end
            end
            assemble!(f, celldofs(facet), fe)
        end
    end

    return f
end

