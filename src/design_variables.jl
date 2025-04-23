struct DesignVariables{T1<:Real,T2<:Real,TV<:AbstractVector{T1},TL<:AbstractVector{T2}} <: AbstractVector{T1}
    variables::TV
    vars_per_element::Int
    lim_inf::TL
    lim_sup::TL
end
DesignVariables(vars_per_element::Int, ::Type{T}) where {T} = DesignVariables(T[], vars_per_element, T[], T[])
DesignVariables(vars_per_element::Int) = DesignVariables(vars_per_element, Float64)

Base.size(x::DesignVariables) = size(x.variables)
Base.IndexStyle(::Type{<:DesignVariables}) = IndexLinear()
Base.getindex(x::DesignVariables, i::Int) = x.variables[i]
Base.setindex!(x::DesignVariables, xe, i::Int) = (x.variables[i] = xe)
Base.similar(x::DesignVariables, ::Type{S}, dims::Dims) where {S} = DesignVariables(similar(x.variables, S, dims), x.vars_per_element, x.lim_inf, x.lim_sup)

function Base.push!(x::DesignVariables, xe, lim_inf, lim_sup)
    push!(x.variables, xe)
    push!(x.lim_inf, lim_inf)
    push!(x.lim_sup, lim_sup)
end

function element_slice(x::DesignVariables, el::Int)
    n = x.vars_per_element
    return @view x[n*(el-1)+1:n*el]
end
function parameter_slice(x::DesignVariables, idx::Int)
    n = x.vars_per_element
    return @view x[idx:n:end]
end

get_limits(x::DesignVariables, i::Int) = (x.lim_inf[i], x.lim_sup[i])

# applying broadcast only on the variables field
# magic from https://docs.julialang.org/en/v1/manual/interfaces/
Base.BroadcastStyle(::Type{<:DesignVariables}) = Broadcast.ArrayStyle{DesignVariables}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{DesignVariables}}, ::Type{ElType}) where {ElType}
    x = find_dv(bc)
    DesignVariables(similar(Array{ElType}, axes(bc)), x.vars_per_element, x.lim_inf, x.lim_sup)
end
find_dv(bc::Base.Broadcast.Broadcasted) = find_dv(bc.args)
find_dv(args::Tuple) = find_dv(find_dv(args[1]), Base.tail(args))
find_dv(x) = x
find_dv(::Tuple{}) = nothing
find_dv(x::DesignVariables, _) = x
find_dv(::Any, rest) = find_dv(rest)

# define views and multiplication for the filters
Base.view(A::DesignVariables, inds...) = DesignVariables(view(A.variables, inds...), A.vars_per_element, view(A.lim_inf, inds...), view(A.lim_sup, inds...))
LinearAlgebra.mul!(C::DesignVariables, A::AbstractVecOrMat, B::DesignVariables, α::Number, β::Number) = mul!(C.variables, A, B.variables, α, β)
