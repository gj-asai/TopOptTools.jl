abstract type Load{dim,T<:Real} end

"""
Load applied directly to the nodal load vector
Uses the degrees of freedom of the nodeset `nodeset_name`
Given in units of force
"""
struct NodalLoad{dim,T} <: Load{dim,T}
    nodeset_name::String
    F::NTuple{dim,T}
end

"""
Load added to the weak form of the elasticity equation
Uses the degrees of freedom of the facetset `faceset_name`
Given in units of force per distance
"""
struct LinearLoad{dim,T} <: Load{dim,T}
    faceset_name::String
    F::NTuple{dim,T}
end
