struct MMAWorkspace
    opt::Opt
    asyinit::Float64
    asyincr::Float64
    asydecr::Float64
    move::Float64

    low::Vector{Float64}
    upp::Vector{Float64}
    alfa::Vector{Float64}
    beta::Vector{Float64}

    p0::Vector{Float64}
    q0::Vector{Float64}
    p::Matrix{Float64}
    q::Matrix{Float64}
    r::Vector{Float64}

    xsub::Vector{Float64}
    lowsub::Vector{Float64}
    uppsub::Vector{Float64}
end
function MMAWorkspace(m::Int, n::Int; asyinit=0.5, asyincr=1.2, asydecr=0.7, move=0.5)
    opt = Opt(:LD_CCSAQ, n + m + 1)
    xtol_rel!(opt, 1e-5)
    maxeval!(opt, 200)
    maxtime!(opt, 10)
    MMAWorkspace(
        opt, asyinit, asyincr, asydecr, move,
        Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n),
        Vector{Float64}(undef, n), Vector{Float64}(undef, n),
        Matrix{Float64}(undef, m, n), Matrix{Float64}(undef, m, n), Vector{Float64}(undef, m),
        Vector{Float64}(undef, n + m + 1), zeros(n + m + 1), fill(Inf, n + m + 1)
    )
end

# this function uses an algorithm from NLopt to solve the subproblem efficiently
# it is not used for the main optimization because we want control of what happens after a single iteration
function mma_update!(workspace, m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, a0, a, c, d)
    @unpack opt, asyinit, asyincr, asydecr, move = workspace
    @unpack low, upp, alfa, beta = workspace
    @unpack p0, q0, p, q, r = workspace
    @unpack xsub, lowsub, uppsub = workspace

    raa0 = 1e-5
    albefa = 0.1
    copyto!(r, fval)
    fill!(xsub, 1.0)

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

        xsub[j] = 0.5 * (alfa[j] + beta[j])
        lowsub[j] = alfa[j]
        uppsub[j] = beta[j]

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

    # solve subproblem with NLopt
    remove_constraints!(opt)
    lower_bounds!(opt, lowsub)
    upper_bounds!(opt, uppsub)
    min_objective!(opt, (x, g) -> sub_obj(x, g, n, m, low, upp, p0, q0, a0, c, d))
    for i in 1:m
        inequality_constraint!(opt, (x, g) -> sub_cons(x, g, i, n, m, low, upp, p, q, r, a))
    end

    optimize!(opt, xsub)
    return @view(xsub[1:n])
end

function sub_obj(in::Vector, grad::Vector, n, m, low, upp, p0, q0, a0, c, d)
    x = @view(in[1:n])
    y = @view(in[n+1:n+m])
    z = in[end]

    f0 = 0.0
    # x
    for j in 1:n
        ux = upp[j] - x[j]
        xl = x[j] - low[j]
        f0 += p0[j] / ux + q0[j] / xl
        grad[j] = p0[j] / ux^2 - q0[j] / xl^2
    end
    # y
    for i in 1:m
        f0 += c[i] * y[i] + 0.5 * d[i] * y[i]^2
        grad[n+i] = c[i] + d[i] * y[i]
    end
    # z
    f0 += a0 * z
    grad[end] = a0

    return f0
end

function sub_cons(in::Vector, grad::Vector, i, n, m, low, upp, p, q, r, a)
    x = @view(in[1:n])
    y = @view(in[n+1:n+m])
    z = in[end]

    f = r[i]
    # x
    for j in 1:n
        ux = upp[j] - x[j]
        xl = x[j] - low[j]
        f += p[i, j] / ux + q[i, j] / xl
        grad[j] = p[i, j] / ux^2 - q[i, j] / xl^2
    end
    # y
    @views fill!(grad[n+1:n+m], 0.0)
    f -= y[i]
    grad[n+i] = -1.0
    # z
    f -= a[i] * z
    grad[end] = -a[i]

    return f
end
