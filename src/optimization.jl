@kwdef struct OptimOpts
    maxiter::Integer = 200
    reltol::Real = 1e-3
    s_init::Real = 0.1
    s_decr::Real = 0.6
    s_incr::Real = 1.1
end

function topopt(
    objective::Function, dobjective::Function,
    constraint::Function, dconstraint::Function,
    x0, xmin, xmax,
    model::FEModel, opts::OptimOpts,
    post::Function, # TODO: implement a default value that just returns nothing
)
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
            tol=Tolerance(frel=opts.reltol),
            s_init=opts.s_init,
            s_decr=opts.s_decr,
            s_incr=opts.s_incr,
        ),
        callback=post,
    )
    return r.minimizer
end
