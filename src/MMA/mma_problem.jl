struct MMAProblem{T<:Real,TF<:Objective,TG<:Constraints}
    approx::ConvexApproximation{T}
    state::MMAState{T}
    primal_dual::PrimalDualState{T}
    artificial::ArtificialParameters{T}
    f::TF
    g::TG
end
function MMAProblem(x0::DesignVariables{T}, obj::Objective, cons::Constraints; move=0.5, asyinit=0.5, asydecr=0.7, asyincr=1.2) where {T}
    n = length(x0)
    m = cons.m
    return MMAProblem(
        ConvexApproximation(n, m, move, asyinit, asydecr, asyincr),
        MMAState(x0, obj(x0)..., cons(x0)...),
        PrimalDualState(length(x0), m, T),
        ArtificialParameters(m, T),
        obj, cons,
    )
end

function iterate(prob::MMAProblem)
    if prob.state.it == 0
        reset_timer!(TopOpt.timer)
    end

    @timeit TopOpt.timer "mma" begin
        update_convex_approximation!(prob.approx, prob.state)
        solve_subproblem!(prob.primal_dual, prob.approx, prob.artificial)

        # update state
        state = prob.state
        state.it += 1

        state.xprev2 .= state.xprev1
        state.xprev1 .= state.x
        state.x .= prob.primal_dual.x

        state.cur_obj, state.cur_dobj = prob.f(state.x)
        state.cur_cons, state.cur_dcons = prob.g(state.x)
        push!(state.obj_hist, state.cur_obj)
    end
end
