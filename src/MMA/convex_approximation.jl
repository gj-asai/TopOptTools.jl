function update_asymptotes!(prob::MMAProblem)
    @unpack L, U, α, β, move, asyinit, asydecr, asyincr = prob.approx
    @unpack it, xprev1, xprev2 = prob.state
    x = prob.state.x
    x_range = x.lim_sup - x.lim_inf

    @tasks for j in 1:prob.n
        if it < 3
            L[j] = x[j] - asyinit * x_range[j]
            U[j] = x[j] + asyinit * x_range[j]
        else
            γ = 1.0
            variation = (x[j] - xprev1[j]) * (xprev1[j] - xprev2[j])
            if variation < 0
                γ = asydecr
            elseif variation > 0
                γ = asyincr
            end

            L[j] = x[j] - γ * (xprev1[j] - L[j])
            U[j] = x[j] + γ * (U[j] - xprev1[j])

            L[j] = max(x[j] - 10 * x_range[j], min(L[j], x[j] - 0.01 * x_range[j]))
            U[j] = max(x[j] + 0.01 * x_range[j], min(U[j], x[j] + 10 * x_range[j]))
        end

        α[j] = max(x.lim_inf[j], L[j] + 0.1 * (x[j] - L[j]), x[j] - move * x_range[j])
        β[j] = min(x.lim_sup[j], U[j] - 0.1 * (U[j] - x[j]), x[j] + move * x_range[j])
    end
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

    @tasks for j in 1:prob.n
        p0[j] = (U[j] - xref[j])^2 * (1.001 * df_plus[j] + 0.001 * df_minus[j] + 1e-5 / x_range[j])
        q0[j] = (xref[j] - L[j])^2 * (0.001 * df_plus[j] + 1.001 * df_minus[j] + 1e-5 / x_range[j])
        for i in 1:prob.m
            p[i, j] = (U[j] - xref[j])^2 * (1.001 * dg_plus[i, j] + 0.001 * dg_minus[i, j] + 1e-5 / x_range[j])
            q[i, j] = (xref[j] - L[j])^2 * (0.001 * dg_plus[i, j] + 1.001 * dg_minus[i, j] + 1e-5 / x_range[j])
        end
    end

    for i in 1:prob.m
        r[i] = cur_cons[i] - treduce(+, collect(p[i, j] / (U[j] - xref[j]) + q[i, j] / (xref[j] - L[j]) for j in 1:prob.n))
    end
end
