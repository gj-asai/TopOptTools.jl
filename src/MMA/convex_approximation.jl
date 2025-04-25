function update_asymptotes!(prob::MMAProblem)
    @unpack L, U, α, β, move, asyinit, asydecr, asyincr = prob.approx
    @unpack it, xprev1, xprev2 = prob.state
    x = prob.state.x
    x_range = x.lim_sup - x.lim_inf

    if it < 3
        @. L = x - asyinit * x_range
        @. U = x + asyinit * x_range
    else
        variation = @. (x - xprev1) * (xprev1 - xprev2)
        γ = ones(prob.n)
        @. γ[variation<0] = asydecr
        @. γ[variation>0] = asyincr

        @. L = x - γ * (xprev1 - L)
        @. U = x + γ * (U - xprev1)

        @. L = max(x - 10 * x_range, min(L, x - 0.01 * x_range))
        @. U = max(x + 0.01 * x_range, min(U, x + 10 * x_range))
    end

    @. α = max(x.lim_inf, L + 0.1 * (x - L), x - move * x_range)
    @. β = min(x.lim_sup, U - 0.1 * (U - x), x + move * x_range)
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

    r .= cur_cons
    for j in 1:prob.n
        p0[j] = (U[j] - xref[j])^2 * (1.001 * df_plus[j] + 0.001 * df_minus[j] + 1e-5 / x_range[j])
        q0[j] = (xref[j] - L[j])^2 * (0.001 * df_plus[j] + 1.001 * df_minus[j] + 1e-5 / x_range[j])
        for i in 1:prob.m
            p[i, j] = (U[j] - xref[j])^2 * (1.001 * dg_plus[i, j] + 0.001 * dg_minus[i, j] + 1e-5 / x_range[j])
            q[i, j] = (xref[j] - L[j])^2 * (0.001 * dg_plus[i, j] + 1.001 * dg_minus[i, j] + 1e-5 / x_range[j])
            r[i] -= p[i, j] / (U[j] - xref[j]) + q[i, j] / (xref[j] - L[j])
        end
    end
end
