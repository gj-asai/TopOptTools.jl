"""
Preallocates all memory used by the MMA solver
"""
mutable struct MMAWorkspace
    asyinit::Float64
    asyincr::Float64
    asydecr::Float64
    move::Float64

    a0::Float64
    a::Vector{Float64}
    c::Vector{Float64}
    d::Vector{Float64}
    m::Int
    n::Int

    low::Vector{Float64}
    upp::Vector{Float64}
    alfa::Vector{Float64}
    beta::Vector{Float64}

    p0::Vector{Float64}
    q0::Vector{Float64}
    p::Matrix{Float64}
    q::Matrix{Float64}
    r::Vector{Float64}

    xmin::Vector{Float64}
    xmax::Vector{Float64}
    xold1::Vector{Float64}
    xold2::Vector{Float64}

    primal_x::Vector{Float64}
    primal_y::Vector{Float64}
    primal_z::Float64
    plam::Vector{Float64}
    qlam::Vector{Float64}
    cur_lambda::Vector{Float64}

    iter::Int
end

"""
    MMAWorkspace(m::Int, n::Int, a0, a, c, d; asyinit=0.5, asyincr=1.2, asydecr=0.7, move=0.5)

Prepares an optimization with `m` constraints and `n` design variables.

`a0`, `a`, `c` and `d` are MMA parameters
"""
function MMAWorkspace(m::Int, n::Int, xmin, xmax, a0, a, c, d; asyinit=0.5, asyincr=1.2, asydecr=0.7, move=0.5)
    MMAWorkspace(
        asyinit, asyincr, asydecr, move, a0, a, c, d, m, n,
        Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n), # low upp alfa beta
        Vector{Float64}(undef, n), Vector{Float64}(undef, n), # p0 q0
        Matrix{Float64}(undef, m, n), Matrix{Float64}(undef, m, n), Vector{Float64}(undef, m), # p q r
        xmin, xmax, Vector{Float64}(undef, n), Vector{Float64}(undef, n), # xmin xmax xold1 xold2
        Vector{Float64}(undef, n), Vector{Float64}(undef, m), 0.0, # primal_x primal_y primal_z
        Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, m), # plam qlam cur_lambda
        1, # iter
    )
end

"""
    mma_update!(workspace, x, dobjdx, cons, dconsdx)

Returns the updated value of the design variables, solving the MMA subproblem with a dual method.

`x` must have size `(n,)`, `dobjdx` must have size `(n,)`, `cons` must have size `(m,)`, and `dconsdx` must have size `(m, n)`
"""
function mma_update!(workspace, x, dobjdx, cons, dconsdx)
    @unpack a0, a, c, d, asyinit, asyincr, asydecr, move, m, n, iter = workspace
    @unpack low, upp, alfa, beta = workspace
    @unpack p0, q0, p, q, r = workspace
    @unpack xmin, xmax, xold1, xold2 = workspace
    @unpack primal_x, primal_y, primal_z, plam, qlam, cur_lambda = workspace

    raa0 = 1e-5
    albefa = 0.1
    copyto!(r, cons)

    # update asymptotes
    # no need to do this in parallel because it is much cheaper than solving the subproblem
    for j in 1:n
        xmaxmin = xmax[j] - xmin[j]

        if iter < 2.5
            low[j] = x[j] - asyinit * xmaxmin
            upp[j] = x[j] + asyinit * xmaxmin
        else
            gamma = 1.0
            variation = (x[j] - xold1[j]) * (xold1[j] - xold2[j])
            if variation < 0
                gamma = asydecr
            elseif variation > 0
                gamma = asyincr
            end

            low[j] = x[j] - gamma * (xold1[j] - low[j])
            upp[j] = x[j] + gamma * (upp[j] - xold1[j])

            low[j] = max(x[j] - 10 * xmaxmin, min(low[j], x[j] - 0.01 * xmaxmin))
            upp[j] = max(x[j] + 0.01 * xmaxmin, min(upp[j], x[j] + 10 * xmaxmin))
        end

        ux = upp[j] - x[j]
        xl = x[j] - low[j]

        alfa[j] = max(xmin[j], low[j] + albefa * xl, x[j] - move * xmaxmin)
        beta[j] = min(xmax[j], upp[j] - albefa * ux, x[j] + move * xmaxmin)

        df_plus = max(dobjdx[j], 0)
        df_minus = max(-dobjdx[j], 0)

        p0[j] = ux^2 * (1.001 * df_plus + 0.001 * df_minus + raa0 / xmaxmin)
        q0[j] = xl^2 * (0.001 * df_plus + 1.001 * df_minus + raa0 / xmaxmin)
        for i in 1:m
            dg_plus = max(dconsdx[i, j], 0)
            dg_minus = max(-dconsdx[i, j], 0)

            p[i, j] = ux^2 * (1.001 * dg_plus + 0.001 * dg_minus + raa0 / xmaxmin)
            q[i, j] = xl^2 * (0.001 * dg_plus + 1.001 * dg_minus + raa0 / xmaxmin)
            r[i] -= p[i, j] / ux + q[i, j] / xl
        end
    end

    # recover primal variables from the dual variables
    function update_primal(lambda)
        cur_lambda == lambda && return

        # plam .= sqrt.(p0 + p' * lambda)
        mul!(plam, p', lambda)
        @. plam = sqrt(p0 + plam)

        # qlam .= sqrt.(q0 + q' * lambda)
        mul!(qlam, q', lambda)
        @. qlam = sqrt(q0 + qlam)

        @. primal_x = max(alfa, min((plam * low + qlam * upp) / (plam + qlam), beta))
        @. primal_y = max(0, lambda - c)
        workspace.primal_z = max(0, lambda ⋅ a - 1.0)
        cur_lambda .= lambda
    end

    # subproblem objective from dual variables
    function dual_obj(lambda)
        update_primal(lambda)
        Lx = sum(p0[j] / (upp[j] - primal_x[j]) + q0[j] / (primal_x[j] - low[j]) for j in 1:n) +
             sum(lambda[i] * (sum((p[i, j] / (upp[j] - primal_x[j]) + q[i, j] / (primal_x[j] - low[j])) for j in 1:n) + r[i]) for i in 1:m)
        Ly = sum(c[i] * primal_y[i] + primal_y[i]^2 / 2 - lambda[i] * primal_y[i] for i in 1:m)
        Lz = primal_z[1] + primal_z[1]^2 / 2 - lambda ⋅ a * primal_z[1]
        return -(Lx + Ly + Lz) # flipping the sign because we want to maximize the objective
    end

    # subproblem objective derivative from dual variables
    function dual_jac!(J, lambda)
        update_primal(lambda)
        for i in 1:m
            # TODO: part of this calculations are repeated from dual_obj
            J[i] = sum((p[i, j] / (upp[j] - primal_x[j]) + q[i, j] / (primal_x[j] - low[j])) for j in 1:n) +
                   r[i] - a[i] * primal_z[1] - primal_y[i]
        end
        J .*= -1.0 # because we flipped the sign of the objective
    end

    # solve the subproblem
    optimize(dual_obj, dual_jac!, zeros(m), fill(Inf, m), ones(m))

    # update state
    workspace.iter += 1
    xold2 .= xold1
    xold1 .= x

    return primal_x
end

function restart!(workspace::MMAWorkspace)
    workspace.iter = 1
end
