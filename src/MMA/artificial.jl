struct ArtificialParameters{T<:Real}
    a0::T
    a::Vector{T}
    c::Vector{T}
    d::Vector{T}
end
ArtificialParameters(m::Int, ::Type{T}) where {T} = ArtificialParameters(1.0, zeros(T, m), fill(1e4, m), ones(T, m))
