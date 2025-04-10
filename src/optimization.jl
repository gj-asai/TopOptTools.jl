@kwdef struct OptimOpts
    maxiter::Int = 200
    reltol::Float64 = 1e-3
    s_init::Float64 = 0.2
    s_decr::Float64 = 0.7
    s_incr::Float64 = 0.8
end

const timer = TimerOutput()

function topopt(
    objective::Function, dobjective::Function,
    constraint::Function, dconstraint::Function,
    x0, xmin, xmax,
    model::FEModel, opts::OptimOpts,
    post::Function, # TODO: implement a default value that just returns nothing
)
    reset_timer!(timer)
    x0 = copy(x0)

    mma = Model()
    for (low, upp) in zip(xmin, xmax)
        addvar!(mma, low, upp)
    end

    set_objective!(mma, CustomGradFunction(objective, dobjective))
    add_ineq_constraint!(mma, CustomGradFunction(constraint, dconstraint))

    @info "Starting optimization with p = $(model.mat_interp.penal)"
    r = optimize(
        mma,
        MMA(),
        x0,
        options=MMAOptions(
            maxiter=opts.maxiter,
            convcriteria=GenericCriteria(),
            tol=Tolerance(x=0.0, fabs=0.0, frel=opts.reltol),
            s_init=opts.s_init,
            s_decr=opts.s_decr,
            s_incr=opts.s_incr,
        ),
        callback=post,
    )

    return r.minimizer
end
