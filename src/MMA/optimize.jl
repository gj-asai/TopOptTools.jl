function optimize(x0::DesignVariables, obj::Objective, cons::Constraints)
    prob = MMAProblem(x0, obj, cons)

    for _ = 1:100
        @timeit TopOpt.timer "mma" begin
            update_convex_approximation!(prob)
            solve_subproblem!(prob)
        end
        update_state!(prob)

        relative_change(prob.state) < 1e-8 && break
    end

    return prob.state.x, prob.state.cur_obj
end

function solve_subproblem!(prob::MMAProblem)
    @unpack approx, state, artificial = prob
    @unpack L, U, α, β, p0, q0, p, q, r = approx
    @unpack x = state
    primal_dual = prob.primal_dual

    reset_primal_dual!(prob)
    norm_inf_residual = Inf
    while primal_dual.ε > 1e-7
        for _ = 1:200
            norm_inf_residual < 0.9 * primal_dual.ε && break
            Δw = newton_direction(prob)
            residual = step_primal_dual!(prob, Δw)
            norm_inf_residual = norm(residual, Inf)
        end
        primal_dual.ε *= 0.1
    end
end

function update_state!(prob::MMAProblem)
    xnew = prob.primal_dual.x
    state = prob.state

    state.it += 1

    state.xprev2 .= state.xprev1
    state.xprev1 .= state.x
    state.x .= xnew

    state.prev_obj = state.cur_obj
    state.cur_obj, state.cur_dobj = prob.f(state.x)
    state.cur_cons, state.cur_dcons = prob.g(state.x)
    @info "It = $(state.it), obj = $(state.cur_obj)"
end

relative_change(state::MMAState) = abs(state.cur_obj - state.prev_obj) / state.prev_obj
