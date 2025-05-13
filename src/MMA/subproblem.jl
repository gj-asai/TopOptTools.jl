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

function solve_subproblem!(primal_dual::PrimalDualState, approx::ConvexApproximation, artificial::ArtificialParameters)
    @unpack L, U, α, β, p0, q0, p, q, r = approx

    reset_primal_dual!(primal_dual, approx, artificial)
    while primal_dual.ε > 1e-7
        norm_inf_residual = Inf
        for _ = 1:200
            norm_inf_residual < 0.9 * primal_dual.ε && break
            Δw = newton_direction(primal_dual, approx, artificial)
            new_residual = step_primal_dual!(primal_dual, approx, artificial, Δw)
            norm_inf_residual = norm(new_residual, Inf)
        end
        primal_dual.ε *= 0.1
    end
end

function reset_primal_dual!(primal_dual::PrimalDualState, approx::ConvexApproximation, artificial::ArtificialParameters)
    @unpack α, β = approx
    c = artificial.c

    primal_dual.ε = 1.0
    @. primal_dual.x = 0.5 * (α + β)
    fill!(primal_dual.y, 1.0)
    primal_dual.z = 1.0
    fill!(primal_dual.λ, 1.0)
    @. primal_dual.ξ = max(1.0, 1 / (primal_dual.x - α))
    @. primal_dual.η = max(1.0, 1 / (β - primal_dual.x))
    @. primal_dual.μ = max(1.0, c / 2)
    primal_dual.ζ = 1.0
    fill!(primal_dual.s, 1.0)

    nothing
end

function move_primal_dual!(state::PrimalDualState, step, direction)
    Δx, Δy, Δz, Δλ, Δξ, Δη, Δμ, Δζ, Δs = direction

    @. state.x += step * Δx
    @. state.y += step * Δy
    state.z += step * Δz
    @. state.λ += step * Δλ
    @. state.ξ += step * Δξ
    @. state.η += step * Δη
    @. state.μ += step * Δμ
    state.ζ += step * Δζ
    @. state.s += step * Δs

    nothing
end

function update_kkt_residuals!(residuals, primal_dual::PrimalDualState, approx::ConvexApproximation, artificial::ArtificialParameters)
    @unpack L, U, α, β, p0, q0, p, q, r, m, n = approx
    @unpack a0, a, c, d = artificial
    @unpack ε, x, y, z, λ, ξ, η, μ, ζ, s = primal_dual

    g = zeros(m)
    pλ = copy(p0)
    qλ = copy(q0)
    @inbounds for i in axes(p, 1), j in axes(p, 2)
        g[i] += p[i, j] / (U[j] - x[j]) + q[i, j] / (x[j] - L[j])
        pλ[j] += λ[i] * p[i, j]
        qλ[j] += λ[i] * q[i, j]
    end

    @. residuals[1:n] = pλ / (U - x)^2 - qλ / (x - L)^2 - ξ + η
    @. residuals[n+1:n+m] = c + d * y - λ - μ
    residuals[n+m+1] = a0 - ζ - λ ⋅ a
    @. residuals[n+m+2:n+2m+1] = g - a * z - y + s + r
    @. residuals[n+2m+2:2n+2m+1] = ξ * (x - α) - ε
    @. residuals[2n+2m+2:3n+2m+1] = η * (β - x) - ε
    @. residuals[3n+2m+2:3n+3m+1] = μ * y - ε
    residuals[3n+3m+2] = ζ * z - ε
    @. residuals[3n+3m+3:3n+4m+2] = λ * s - ε

    nothing
end

function newton_direction(primal_dual::PrimalDualState, approx::ConvexApproximation, artificial::ArtificialParameters)
    @unpack L, U, α, β, p0, q0, p, q, r, m, n = approx
    @unpack ε, x, y, z, λ, ξ, η, μ, ζ, s = primal_dual
    @unpack a0, a, c, d = artificial

    g = zeros(m)
    pλ = copy(p0)
    qλ = copy(q0)
    G = zeros(m, n)
    @inbounds for i in axes(p, 1), j in axes(p, 2)
        g[i] += p[i, j] / (U[j] - x[j]) + q[i, j] / (x[j] - L[j])
        pλ[j] += λ[i] * p[i, j]
        qλ[j] += λ[i] * q[i, j]
        G[i, j] = p[i, j] / (U[j] - x[j])^2 - q[i, j] / (x[j] - L[j])^2
    end

    Dx = Diagonal(@. 2 * pλ / (U - x)^3 + 2 * qλ / (x - L)^3 + ξ / (x - α) + η / (β - x))
    Dy = Diagonal(@. d + μ / y)
    Dλ = Diagonal(s ./ λ)
    δx_ = @. pλ / (U - x)^2 - qλ / (x - L)^2 - ε / (x - α) + ε / (β - x)
    δy_ = @. c + d * y - λ - ε / y
    δz_ = a0 - λ ⋅ a - ε / z
    δλ_ = @. g - a * z - y + r + ε / λ

    Dλy = Dλ + inv(Dy)
    δλy_ = δλ_ + inv(Dy) * δy_

    lhs = [Dλy+G*inv(Dx)*G' a; a' [-ζ / z]]
    rhs = [δλy_ - G * inv(Dx) * δx_; δz_]
    sol = lhs \ rhs
    Δλ = sol[1:end-1]
    Δz = sol[end]

    Δx = -inv(Dx) * G' * Δλ - inv(Dx) * δx_
    Δy = inv(Dy) * Δλ - inv(Dy) * δy_

    Δξ = @. -ξ * Δx / (x - α) - ξ + ε / (x - α)
    Δη = @. η * Δx / (β - x) - η + ε / (β - x)
    Δμ = @. -μ * Δy / y - μ + ε / y
    Δζ = -(ζ / z) * Δz - ζ + ε / z
    Δs = @. -s * Δλ / λ - s + ε / λ

    return Δx, Δy, Δz, Δλ, Δξ, Δη, Δμ, Δζ, Δs
end

function step_primal_dual!(primal_dual::PrimalDualState{T}, approx::ConvexApproximation, artificial::ArtificialParameters, Δw) where {T}
    @unpack ε, x, y, z, λ, ξ, η, μ, ζ, s = primal_dual
    @unpack α, β, n, m = approx
    Δx, Δy, Δz, Δλ, Δξ, Δη, Δμ, Δζ, Δs = Δw

    tα = maximum(@. -1.01 * Δx / (x - α))
    tβ = maximum(@. 1.01 * Δx / (β - x))
    ty = maximum(@. -1.01 * Δy / y)
    tz = -1.01 * Δz / z
    tλ = maximum(@. -1.01 * Δλ / λ)
    tξ = maximum(@. -1.01 * Δξ / ξ)
    tη = maximum(@. -1.01 * Δη / η)
    tμ = maximum(@. -1.01 * Δμ / μ)
    tζ = -1.01 * Δζ / ζ
    ts = maximum(@. -1.01 * Δs / s)
    t = 1.0 / max(tα, tβ, ty, tz, tλ, tξ, tη, tμ, tζ, ts, 1.0)

    residuals = Vector{T}(undef, 3n + 4m + 2)

    update_kkt_residuals!(residuals, primal_dual, approx, artificial)
    norm_initial_residual = norm(residuals)

    move_primal_dual!(primal_dual, t, Δw)
    update_kkt_residuals!(residuals, primal_dual, approx, artificial)
    for _ = 1:50
        norm(residuals) < norm_initial_residual && break

        # if residual increased, walk half way back
        t /= 2
        move_primal_dual!(primal_dual, -t, Δw)
        update_kkt_residuals!(residuals, primal_dual, approx, artificial)
    end

    return residuals
end
