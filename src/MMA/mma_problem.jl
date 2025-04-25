mutable struct MMAState{T<:Real}
    it::Int
    x::DesignVariables{T,T}
    xprev1::DesignVariables{T,T}
    xprev2::DesignVariables{T,T}

    cur_obj::T
    prev_obj::T
    cur_dobj::Vector{T}
    cur_cons::Vector{T}
    cur_dcons::Matrix{T}
end
MMAState(x0, cur_obj, cur_dobj, cur_cons, cur_dcons) = MMAState(0, x0, similar(x0), similar(x0), cur_obj, Inf, cur_dobj, cur_cons, cur_dcons)

struct ConvexApproximation{T<:Real}
    # asymptote move limits
    move::T
    asyinit::T
    asydecr::T
    asyincr::T

    # asymptotes and variable limits
    L::Vector{T}
    U::Vector{T}
    α::Vector{T}
    β::Vector{T}

    # approximation coefficients
    p0::Vector{T}
    q0::Vector{T}
    p::Matrix{T}
    q::Matrix{T}
    r::Vector{T}
end
ConvexApproximation(n::Int, m::Int, opts::OptimOpts, ::Type{T}) where {T<:Real} = ConvexApproximation(
    opts.move, opts.asyinit, opts.asydecr, opts.asyincr,
    Vector{T}(undef, n), # L
    Vector{T}(undef, n), # U
    Vector{T}(undef, n), # α
    Vector{T}(undef, n), # β
    Vector{T}(undef, n), # p0
    Vector{T}(undef, n), # q0
    Matrix{T}(undef, m, n), # p
    Matrix{T}(undef, m, n), # q
    Vector{T}(undef, m), # r
)

struct ArtificialParameters{T<:Real}
    a0::T
    a::Vector{T}
    c::Vector{T}
    d::Vector{T}
end
ArtificialParameters(m::Int, ::Type{T}) where {T} = ArtificialParameters(1.0, zeros(T, m), fill(1e5, m), ones(T, m))

mutable struct PrimalDualState{T<:Real}
    ε::T
    x::Vector{T}
    y::Vector{T}
    z::T
    λ::Vector{T}
    ξ::Vector{T}
    η::Vector{T}
    μ::Vector{T}
    ζ::T
    s::Vector{T}
end
PrimalDualState(n::Int, m::Int, ::Type{T}) where {T} = PrimalDualState(
    zero(T),
    Vector{T}(undef, n),
    Vector{T}(undef, m),
    zero(T),
    Vector{T}(undef, m),
    Vector{T}(undef, n),
    Vector{T}(undef, n),
    Vector{T}(undef, m),
    zero(T),
    Vector{T}(undef, m),
)

struct MMAProblem{T<:Real,TF<:Objective,TG<:Constraints}
    approx::ConvexApproximation{T}
    state::MMAState{T}
    primal_dual::PrimalDualState{T}
    residuals::Vector{T}
    artificial::ArtificialParameters{T}
    f::TF
    g::TG
    n::Int
    m::Int
end
function MMAProblem(x0::DesignVariables{T}, obj::Objective, cons::Constraints, opts::OptimOpts=OptimOpts()) where {T}
    n = length(x0)
    m = cons.m
    return MMAProblem(
        ConvexApproximation(n, m, opts, T),
        MMAState(x0, obj(x0)..., cons(x0)...),
        PrimalDualState(length(x0), m, T),
        Vector{T}(undef, 3n + 4m + 2),
        ArtificialParameters(m, T),
        obj, cons, n, m
    )
end
