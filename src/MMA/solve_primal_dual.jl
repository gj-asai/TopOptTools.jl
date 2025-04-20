function reset_primal_dual!(problem::MMAProblem)
    @unpack α, β = problem.approx
    c = problem.artificial.c
    state = problem.primal_dual

    state.ε = 1.0
    @. state.x = 0.5 * (α + β)
    fill!(state.y, 1.0)
    state.z = 1.0
    fill!(state.λ, 1.0)
    @. state.ξ = max(1.0, 1 / (state.x - α))
    @. state.η = max(1.0, 1 / (β - state.x))
    @. state.μ = max(1.0, c / 2)
    state.ζ = 1.0
    fill!(state.s, 1.0)
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
end

function primal_dual_kkt_residuals(problem::MMAProblem{T}) where {T}
    @unpack approx, artificial, primal_dual = problem
    @unpack L, U, α, β, p0, q0, p, q, r = approx
    @unpack a0, a, c, d = artificial
    @unpack ε, x, y, z, λ, ξ, η, μ, ζ, s = primal_dual

    g = sum(p ./ (U .- x)' + q ./ (x .- L)', dims=2) |> vec
    pλ = p0 + p' * λ
    qλ = q0 + q' * λ

    ∂ψ = @. pλ / (U - x)^2 - qλ / (x - L)^2

    residuals = T[]
    append!(residuals, ∂ψ .- ξ .+ η)
    append!(residuals, c .+ d .* y .- λ .- μ)
    push!(residuals, a0 - ζ - λ ⋅ a)
    append!(residuals, g .- a .* z .- y .+ s .+ r)
    append!(residuals, ξ .* (x .- α) .- ε)
    append!(residuals, η .* (β .- x) .- ε)
    append!(residuals, μ .* y .- ε)
    push!(residuals, ζ * z - ε)
    append!(residuals, λ .* s .- ε)

    return residuals
end

function newton_direction(problem::MMAProblem)
    @unpack approx, primal_dual, artificial = problem
    @unpack L, U, α, β, p0, q0, p, q, r = approx
    @unpack ε, x, y, z, λ, ξ, η, μ, ζ, s = primal_dual
    @unpack a0, a, c, d = artificial

    g = sum(p ./ (U .- x)' + q ./ (x .- L)', dims=2) |> vec
    pλ = p0 + p' * λ
    qλ = q0 + q' * λ

    ∂ψ = @. pλ / (U - x)^2 - qλ / (x - L)^2
    Ψ = Diagonal(@. 2 * pλ / (U - x)^3 + 2 * qλ / (x - L)^3)
    G = @. p / (U - x)'^2 - q / (x - L)'^2

    Dx = Ψ + Diagonal(@. ξ / (x - α) + η / (β - x))
    Dy = Diagonal(@. d + μ / y)
    Dλ = Diagonal(s ./ λ)
    δx_ = @. ∂ψ - ε / (x - α) + ε / (β - x)
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

function step_primal_dual!(problem::MMAProblem, Δw)
    @unpack ε, x, y, z, λ, ξ, η, μ, ζ, s = problem.primal_dual
    @unpack α, β = problem.approx
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

    initial_residual = primal_dual_kkt_residuals(problem)
    norm_initial_residual = norm(initial_residual)

    move_primal_dual!(problem.primal_dual, t, Δw)
    new_residual = primal_dual_kkt_residuals(problem)
    for _ = 1:50
        norm(new_residual) < norm_initial_residual && break

        # if residual increased, walk half way back
        t /= 2
        move_primal_dual!(problem.primal_dual, -t, Δw)
        new_residual = primal_dual_kkt_residuals(problem)
    end

    return new_residual
end
