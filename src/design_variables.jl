"""
    DesignVector(vars_per_element::Int, ::Type{T})
    DesignVector(vars_per_element::Int)

`AbstractVector` to be used as input for the optimization.
It is initialized empty, use [`push!`](@ref) to populate it

`vars_per_element` is the number of design variables per finite element.
It is used during FE preparation to split the vector for [`interpolate`](@ref).

The struct also contains the lower and upper limits of each design variable for MMA
"""
struct DesignVector{T1<:Real,T2<:Real,TV<:AbstractVector{T1},TL<:AbstractVector{T2}} <: AbstractVector{T1}
    variables::TV
    vars_per_element::Int
    lim_inf::TL
    lim_sup::TL
end
DesignVector(vars_per_element::Int, ::Type{T}) where {T} = DesignVector(T[], vars_per_element, T[], T[])
DesignVector(vars_per_element::Int) = DesignVector(vars_per_element, Float64)

# TODO: improve the initialization of the DesignVector
"""
    Base.push!(x::DesignVector, xe, lim_inf, lim_sup)

Pushes a new element to `x` with initial value `xe` and limits `lim_inf` and `lim_sup`
"""
function Base.push!(x::DesignVector, xe, lim_inf, lim_sup)
    push!(x.variables, xe)
    push!(x.lim_inf, lim_inf)
    push!(x.lim_sup, lim_sup)
end

"""
    element_slice(x::DesignVector, el::Int)

Returns a view of `x` that contains the design variables relative to the element number `el`
"""
function element_slice(x::DesignVector, el::Int)
    n = x.vars_per_element
    return @view x[n*(el-1)+1:n*el]
end

"""
    parameter_slice(x::DesignVector, idx::Int)

Returns a view of `x` that contains the `idx`-th design variable of each element
"""
function parameter_slice(x::DesignVector, idx::Int)
    n = x.vars_per_element
    return @view x[idx:n:end]
end

# if not a DesignVector, assume there is one design variable per element
element_slice(x::AbstractArray, el::Int) = @view x[el:el]
parameter_slice(x::AbstractArray, ::Int) = x

"""
    get_limits(x::DesignVector, i::Int)

Returns a `Tuple` with the lower and upper limits for the `i`-th design variable
"""
get_limits(x::DesignVector, i::Int) = (x.lim_inf[i], x.lim_sup[i])

# define AbstractVector behavior 
Base.size(x::DesignVector) = size(x.variables)
Base.IndexStyle(::Type{<:DesignVector}) = IndexLinear()
Base.getindex(x::DesignVector, i::Int) = x.variables[i]
Base.setindex!(x::DesignVector, xe, i::Int) = (x.variables[i] = xe)
Base.similar(x::DesignVector, ::Type{S}, dims::Dims) where {S} = DesignVector(similar(x.variables, S, dims), x.vars_per_element, x.lim_inf, x.lim_sup)

# apply broadcast only on the variables field
# magic from https://docs.julialang.org/en/v1/manual/interfaces/
Base.BroadcastStyle(::Type{<:DesignVector}) = Broadcast.ArrayStyle{DesignVector}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{DesignVector}}, ::Type{ElType}) where {ElType}
    x = find_dv(bc)
    DesignVector(similar(Array{ElType}, axes(bc)), x.vars_per_element, x.lim_inf, x.lim_sup)
end
find_dv(bc::Base.Broadcast.Broadcasted) = find_dv(bc.args)
find_dv(args::Tuple) = find_dv(find_dv(args[1]), Base.tail(args))
find_dv(x) = x
find_dv(::Tuple{}) = nothing
find_dv(x::DesignVector, _) = x
find_dv(::Any, rest) = find_dv(rest)

# define views and multiplication for the filters
Base.view(A::DesignVector, inds...) = DesignVector(view(A.variables, inds...), A.vars_per_element, view(A.lim_inf, inds...), view(A.lim_sup, inds...))
LinearAlgebra.mul!(C::DesignVector, A::AbstractVecOrMat, B::DesignVector, α::Number, β::Number) = mul!(C.variables, A, B.variables, α, β)
