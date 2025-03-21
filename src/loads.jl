abstract type Load{dim,T<:Real} end

struct NodalLoad{dim,T} <: Load{dim,T}
    nodeset_name::String
    F::NTuple{dim,T}
end

struct LinearLoad{dim,T} <: Load{dim,T}
    faceset_name::String
    F::NTuple{dim,T}
end
