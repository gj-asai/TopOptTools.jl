function update_asymptotes!(prob::MMAProblem)
    @unpack L, U, α, β = prob.approx
    @unpack it, xprev1, xprev2 = prob.state
    x = prob.state.x
    x_range = x.lim_sup - x.lim_inf

    albefa = 0.1
    move = 0.5
    asyinit = 0.5
    asydecr = 0.7
    asyincr = 1.2

    if it < 3
        @. L = x - asyinit * x_range
        @. U = x + asyinit * x_range
    else
        variation = @. (x - xprev1) * (xprev1 - xprev2)
        γ = ones(prob.n)
        @. γ[variation.<0] = asydecr
        @. γ[variation.>0] = asyincr

        @. L = x - γ * (x - L)
        @. U = x + γ * (U - x)

        @. L = max(x - 10 * x_range, min(L, x - 0.01 * x_range))
        @. U = max(x + 0.01 * x_range, min(U, x + 10 * x_range))
    end

    @. α = max(x.lim_inf, L + albefa * (x - L), x - move * x_range)
    @. β = min(x.lim_sup, U - albefa * (U - x), x + move * x_range)
end

function update_convex_approximation!(prob::MMAProblem)
    update_asymptotes!(prob)

    @unpack cur_dobj, cur_cons, cur_dcons = prob.state
    @unpack L, U, p0, q0, p, q, r = prob.approx
    xref = prob.state.x
    x_range = xref.lim_sup - xref.lim_inf

    df_plus = max.(cur_dobj, 0)
    df_minus = max.(-cur_dobj, 0)

    dg_plus = max.(cur_dcons, 0)
    dg_minus = max.(-cur_dcons, 0)

    @. p0 = (U - xref)^2 * (1.001 * df_plus + 0.001 * df_minus + 1e-5 / x_range)
    @. q0 = (xref - L)^2 * (0.001 * df_plus + 1.001 * df_minus + 1e-5 / x_range)
    for i in 1:prob.m
        @. p[i, :] = (U - xref) .^ 2 * (1.001 * dg_plus[i, :] + 0.001 * dg_minus[i, :] + 1e-5 / x_range)
        @. q[i, :] = (xref - L) .^ 2 * (0.001 * dg_plus[i, :] + 1.001 * dg_minus[i, :] + 1e-5 / x_range)
        r[i] = cur_cons[i] - sum(@. p[i, :] / (U - xref) + q[i, :] / (xref - L))
    end
end
