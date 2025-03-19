abstract type Load{dim} end

struct NodalLoad{dim} <: Load{dim}
    nodeset_name::AbstractString
    F::NTuple{dim}
end

struct LinearLoad{dim} <: Load{dim}
    faceset_name::AbstractString
    F::NTuple{dim}
end
