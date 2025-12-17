abstract type Load{dim} end
abstract type DesignIndependentLoad{dim} <: Load{dim} end
abstract type DesignDependentLoad{dim} <: Load{dim} end

"""
Load applied directly to the nodal load vector.
Uses the degrees of freedom of the nodeset `nodeset_name`.
Given in units of force
"""
struct NodalLoad{dim} <: DesignIndependentLoad{dim}
    nodeset_name::String
    F::NTuple{dim,Float64}
end

"""
Load added to the weak form of the elasticity equation.
Uses the degrees of freedom of the facetset `faceset_name`.
Given in units of force per length
"""
struct LinearLoad{dim} <: DesignIndependentLoad{dim}
    faceset_name::String
    F::NTuple{dim,Float64}
end

# TODO: create BodyLoad

"""
Design dependent load applied to all elements in the domain.
Defined by the function `stress_function(xe)`, which must return a `SymmetricTensor{2,dim}`
containing the eigenstress associated to this load, equal to elasticity tensor : eigenstrain
"""
struct EigenstrainLoad{F<:Function,dim} <: DesignDependentLoad{dim}
    stress_function::F
end
EigenstrainLoad(stress_function::F; dim) where {F} = EigenstrainLoad{F,dim}(stress_function)

"""
    compute_rhs(loads::Vector{<:Load}, model::FEModel, [x::AbstractVector])
    compute_rhs!(rhs::AbstractVector, [jac_rhs::Matrix], loads::Vector{<:Load}, model::FEModel, [x::AbstractVector])

Returns the nodal forces vector corresponding to the load case `loads` after applying the Dirichlet boundary conditions in `model`.
If `loads` contains design dependent loads, they are calculated from `x` and the function also returns the jacobian of `rhs` with respect to `x`.
When calling the mutating method `compute_rhs!` with design dependent loads, both `jac_rhs`and `x` are necessary.
"""
function compute_rhs!(rhs::AbstractVector, loads::Vector{<:Load}, model::FEModel)
    if any([load isa DesignDependentLoad for load in loads])
        @warn "Ignoring design dependent loads. To include them, use compute_rhs!(rhs, loads, model, x) or compute_rhs(loads, model, x)"
    end

    fill!(rhs, 0.0)
    add_design_independent_loads!(rhs, loads, model)
    apply!(rhs, model.ch)
    return rhs
end

function compute_rhs!(rhs::AbstractVector, jac_rhs::Matrix, loads::Vector{<:Load}, model::FEModel, x::AbstractVector)
    fill!(rhs, 0.0)
    add_design_independent_loads!(rhs, loads, model)
    add_design_dependent_loads!(rhs, jac_rhs, loads, model, x)
    apply!(rhs, model.ch)
    return rhs, jac_rhs
end

compute_rhs(loads::Vector{<:Load}, model::FEModel) = compute_rhs!(zeros(ndofs(model.dh)), loads, model)
compute_rhs(loads::Vector{<:Load}, model::FEModel, x::AbstractVector) = compute_rhs!(zeros(ndofs(model.dh)), zeros(ndofs(model.dh), length(x)), loads, model, x)

function add_design_independent_loads!(rhs::AbstractVector, loads::Vector{<:Load}, model::FEModel{dim}) where {dim}
    fe = zeros(getnbasefunctions(model.facetvalues))
    for cell in CellIterator(model.dh)
        dofs = celldofs(cell)

        # nodal forces
        for force in filter(f -> f isa NodalLoad, loads)
            for (i, node) in enumerate(getnodes(cell))
                node in getnodeset(model.grid, force.nodeset_name) || continue
                rhs[dofs[dim*(i-1)+1:dim*i]] .+= force.F
            end
        end

        # linear forces
        for force in filter(f -> f isa LinearLoad, loads)
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
            assemble!(rhs, dofs, fe)
        end
    end
    return rhs
end

# TODO: make this multithreaded?
function add_design_dependent_loads!(rhs::AbstractVector, jac_rhs::Matrix, loads::Vector{<:Load}, model::FEModel{dim,nvar}, x::AbstractVector) where {dim,nvar}
    fe = zeros(getnbasefunctions(model.facetvalues))
    jac = zeros(getnbasefunctions(model.facetvalues), nvar)
    for cell in CellIterator(model.dh)
        e = cellid(cell)
        xe = @view x[nvar*(e-1)+1:nvar*e]
        dofs = celldofs(cell)

        fill!(fe, 0.0)
        for load in filter(f -> f isa EigenstrainLoad, loads)
            function integrate_element(fe, xe)
                @inbounds for q_point in 1:getnquadpoints(model.cellvalues)
                    dΩ = getdetJdV(model.cellvalues, q_point)
                    for i in 1:getnbasefunctions(model.cellvalues)
                        ∇sδεi = shape_symmetric_gradient(model.cellvalues, q_point, i)
                        fe[i] += ∇sδεi ⊡ load.stress_function(xe) * dΩ
                    end
                end
            end

            ForwardDiff.jacobian!(jac, integrate_element, fe, xe)
        end

        assemble!(rhs, dofs, fe)
        jac_rhs[dofs, nvar*(e-1)+1:nvar*e] .= jac
    end
    return rhs
end
