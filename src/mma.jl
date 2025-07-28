"""
Preallocates all memory used by the MMA solver
"""
struct MMAWorkspace{M<:JuMP.Model}
    model::M

    asyinit::Float64
    asyincr::Float64
    asydecr::Float64
    move::Float64

    a0::Float64
    a::Vector{Float64}
    c::Vector{Float64}
    d::Vector{Float64}

    low::Vector{Float64}
    upp::Vector{Float64}
    alfa::Vector{Float64}
    beta::Vector{Float64}

    p0::Vector{Float64}
    q0::Vector{Float64}
    p::Matrix{Float64}
    q::Matrix{Float64}
    r::Vector{Float64}
end

"""
    MMAWorkspace(m::Int, n::Int, a0, a, c, d; asyinit=0.5, asyincr=1.2, asydecr=0.7, move=0.5)

Prepares an optimization with `m` constraints and `n` design variables.

`a0`, `a`, `c` and `d` are MMA parameters
"""
function MMAWorkspace(m::Int, n::Int, a0, a, c, d; asyinit=0.5, asyincr=1.2, asydecr=0.7, move=0.5)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_string_names_on_creation(model, false)
    set_attribute(model, "tol", 1e-2)
    set_attribute(model, "dual_inf_tol", 1e-2)
    set_attribute(model, "constr_viol_tol", 1e-2)
    set_attribute(model, "compl_inf_tol", 1e-2)
    set_attribute(model, "linear_solver", "spral")

    @variable(model, x[1:n])
    @variable(model, 0 <= y[1:m], start = 1.0)
    @variable(model, 0 <= z, start = 1.0)
    @variables(model, begin
        p0[1:n] in Parameter(0.0)
        q0[1:n] in Parameter(0.0)
        p[1:m, 1:n] in Parameter(0.0)
        q[1:m, 1:n] in Parameter(0.0)
        r[1:m] in Parameter(0.0)
        upp[1:n] in Parameter(0.0)
        low[1:n] in Parameter(0.0)
    end)

    @objective(model, Min,
        sum(p0[j] / (upp[j] - x[j]) + q0[j] / (x[j] - low[j]) for j in 1:n) + a0 * z + sum(c[i] * y[i] + 0.5 * d[i] * y[i]^2 for i in 1:m)
    )
    @constraint(model, cons[i = 1:m],
        sum(p[i, j] / (upp[j] - x[j]) + q[i, j] / (x[j] - low[j]) for j in 1:n) - a[i] * z - y[i] + r[i] <= 0
    )

    MMAWorkspace(
        model, asyinit, asyincr, asydecr, move, a0, a, c, d,
        Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n),
        Vector{Float64}(undef, n), Vector{Float64}(undef, n),
        Matrix{Float64}(undef, m, n), Matrix{Float64}(undef, m, n), Vector{Float64}(undef, m),
    )
end

"""
    mma_update!(workspace, m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx)

Returns the updated value of the design variables

The subproblem is defined in a `JuMP` model and solved using `Ipopt`
"""
function mma_update!(workspace, m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx)
    @unpack model, asyinit, asyincr, asydecr, move = workspace
    @unpack low, upp, alfa, beta = workspace
    @unpack p0, q0, p, q, r = workspace

    raa0 = 1e-5
    albefa = 0.1
    copyto!(r, fval)

    for j in 1:n
        xmaxmin = xmax[j] - xmin[j]

        if iter < 2.5
            low[j] = xval[j] - asyinit * xmaxmin
            upp[j] = xval[j] + asyinit * xmaxmin
        else
            gamma = 1.0
            variation = (xval[j] - xold1[j]) * (xold1[j] - xold2[j])
            if variation < 0
                gamma = asydecr
            elseif variation > 0
                gamma = asyincr
            end

            low[j] = xval[j] - gamma * (xold1[j] - low[j])
            upp[j] = xval[j] + gamma * (upp[j] - xold1[j])

            low[j] = max(xval[j] - 10 * xmaxmin, min(low[j], xval[j] - 0.01 * xmaxmin))
            upp[j] = max(xval[j] + 0.01 * xmaxmin, min(upp[j], xval[j] + 10 * xmaxmin))
        end

        ux = upp[j] - xval[j]
        xl = xval[j] - low[j]

        alfa[j] = max(xmin[j], low[j] + albefa * xl, xval[j] - move * xmaxmin)
        beta[j] = min(xmax[j], upp[j] - albefa * ux, xval[j] + move * xmaxmin)

        df_plus = max(df0dx[j], 0)
        df_minus = max(-df0dx[j], 0)

        p0[j] = ux^2 * (1.001 * df_plus + 0.001 * df_minus + raa0 / xmaxmin)
        q0[j] = xl^2 * (0.001 * df_plus + 1.001 * df_minus + raa0 / xmaxmin)
        for i in 1:m
            dg_plus = max(dfdx[i, j], 0)
            dg_minus = max(-dfdx[i, j], 0)

            p[i, j] = ux^2 * (1.001 * dg_plus + 0.001 * dg_minus + raa0 / xmaxmin)
            q[i, j] = xl^2 * (0.001 * dg_plus + 1.001 * dg_minus + raa0 / xmaxmin)
            r[i] -= p[i, j] / ux + q[i, j] / xl
        end
    end

    # update the subproblem model
    for j in 1:n
        set_lower_bound(model[:x][j], alfa[j])
        set_upper_bound(model[:x][j], beta[j])
        set_start_value(model[:x][j], 0.5 * (alfa[j] + beta[j]))
        set_parameter_value(model[:upp][j], upp[j])
        set_parameter_value(model[:low][j], low[j])
        set_parameter_value(model[:p0][j], p0[j])
        set_parameter_value(model[:q0][j], q0[j])
    end
    for i in 1:m
        set_parameter_value(model[:r][i], r[i])
    end
    for i in 1:m, j in 1:n
        set_parameter_value(model[:p][i, j], p[i, j])
        set_parameter_value(model[:q][i, j], q[i, j])
    end

    # solve the subproblem
    optimize!(model)
    assert_is_solved_and_feasible(model) # should only error if the solver was interrupted
    return value(model[:x])
end
