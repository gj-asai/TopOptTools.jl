# task local data for parallel stiffness assemble
struct ScratchData{CC,CV,T,A}
    cell_cache::CC
    cellvalues::CV
    Ke::Matrix{T}
    jac::Matrix{T}
    assembler::A
end

function ScratchData(model::FEModel, K::SparseMatrixCSC)
    cell_cache = CellCache(model.dh)
    cellvalues = copy(model.cellvalues)

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    jac = zeros(n_basefuncs * n_basefuncs, get_nvar(model))

    asm = start_assemble(K; fillzero=false)
    return ScratchData(cell_cache, cellvalues, Ke, jac, asm)
end

struct FEResults{T<:Real,MT<:SparseMatrixCSC{T},SD<:ScratchData}
    K::MT
    f::Vector{T}
    ∂Ke∂x::Vector{Matrix{T}}
    u::Vector{T}
    chnl::Channel{SD}
end

function FEResults(model::FEModel)
    f = compute_force_vector(model)
    apply!(f, model.ch)

    # preallocate stiffness and sensitivities
    n_basefuncs = getnbasefunctions(model.cellvalues)
    K = allocate_matrix(model.dh)
    ∂Ke∂x = fill(zeros(n_basefuncs, n_basefuncs), get_nvar(model) * getncells(model.grid))

    # preallocate solution
    u = zeros(ndofs(model.dh))

    chnl = Channel{ScratchData}(Threads.nthreads())
    foreach(1:Threads.nthreads()) do _
        put!(chnl, ScratchData(model, K))
    end

    return FEResults(K, f, ∂Ke∂x, u, chnl)
end

function fea!(results::FEResults, x::DesignVariables, model::FEModel)
    @unpack K, f, u = results

    @timeit timer "assemble" begin
        global_stiffness!(results, x, model)
        apply!(K, model.ch)
    end

    @timeit timer "solve" begin
        u .= LinearSolve.solve(
            LinearSolve.LinearProblem(K, f),
            SparspakFactorization(),
        ).u
    end
end

function global_stiffness!(results::FEResults, x::DesignVariables, model::FEModel)
    @unpack K, ∂Ke∂x, chnl = results

    n_basefuncs = getnbasefunctions(model.cellvalues)
    nvar = get_nvar(model)

    start_assemble(K) # zero K out
    for color in model.colors
        @tasks for e in color
            scratch = take!(chnl)
            @unpack cell_cache, cellvalues, Ke, jac, assembler = scratch

            Ferrite.reinit!(cell_cache, e)
            Ferrite.reinit!(cellvalues, cell_cache)
            xe = element_slice(x, e)

            ForwardDiff.jacobian!(jac, (Ke, xe) -> element_stiffness!(Ke, xe, cellvalues, model), Ke, xe)
            for var_idx = 1:nvar
                ∂Ke∂x[nvar*(e-1)+var_idx] = reshape(jac[:, var_idx], n_basefuncs, n_basefuncs)
            end

            assemble!(assembler, celldofs(cell_cache), Ke)
            put!(chnl, scratch)
        end
    end
end

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
            Ferrite.reinit!(model.facetvalues, facet)
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

function stress(result::FEResults, x::DesignVariables, model::FEModel{dim}) where {dim}
    @unpack cellvalues, mat_interp, grid, dh = model

    qp_global = [
        [zero(SymmetricTensor{2,dim}) for _ in 1:getnquadpoints(cellvalues)]
        for _ in 1:getncells(grid)
    ]
    qp_material = [
        [zero(SymmetricTensor{2,dim}) for _ in 1:getnquadpoints(cellvalues)]
        for _ in 1:getncells(grid)
    ]
    qp_vonmises = [
        [zero(Float64) for _ in 1:getnquadpoints(cellvalues)]
        for _ in 1:getncells(grid)
    ]
    qp_principal = [
        [zero(SymmetricTensor{2,dim}) for _ in 1:getnquadpoints(cellvalues)]
        for _ in 1:getncells(grid)
    ]
    qp_directions = [
        [zeros(dim, dim) for _ in 1:getnquadpoints(cellvalues)]
        for _ in 1:getncells(grid)
    ]

    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues, cell)
        e = cellid(cell)
        cell_global = qp_global[e]
        cell_material = qp_material[e]
        cell_vonmises = qp_vonmises[e]
        cell_principal = qp_principal[e]
        cell_directions = qp_directions[e]
        for q_point in 1:getnquadpoints(cellvalues)
            xe = element_slice(x, e)
            ε = function_symmetric_gradient(cellvalues, q_point, result.u, celldofs(cell))
            σ = interpolate(xe, mat_interp) ⊡ ε
            s = dev(σ)

            cell_global[q_point] = σ
            cell_material[q_point] = rotate_stress(σ, xe, mat_interp)
            cell_vonmises[q_point] = sqrt(1.5 * s ⊡ s)

            # sort eigenvalues by absolute value, largest first
            principal = eigen(σ)
            principal_order = sortperm(abs.(principal.values), rev=true)
            cell_principal[q_point] = diagm(SymmetricTensor{2,dim}, principal.values[principal_order])
            cell_directions[q_point] = principal.vectors[:, principal_order]
        end
    end

    return qp_global, qp_material, qp_vonmises, qp_principal, qp_directions
end
