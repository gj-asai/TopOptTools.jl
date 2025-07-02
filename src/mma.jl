struct MMAWorkspace
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
MMAWorkspace(m::Int, n::Int; asyinit=0.5, asyincr=1.2, asydecr=0.7, move=0.5) = MMAWorkspace(
    asyinit, asyincr, asydecr, move,
    Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n),
    Vector{Float64}(undef, n), Vector{Float64}(undef, n),
    Matrix{Float64}(undef, m, n), Matrix{Float64}(undef, m, n), Vector{Float64}(undef, m),
    Vector{Float64}(undef, n + m + 1), zeros(n + m + 1), fill(Inf, n + m + 1)
)

# this function uses an algorithm from NLopt to solve the subproblem efficiently
# it is not used for the main optimization because we want control of what happens after a single iteration
function mma_update!(workspace, m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, a0, a, c, d)
    @unpack asyinit, asyincr, asydecr, move = workspace
    @unpack low, upp, alfa, beta = workspace
    @unpack p0, q0, p, q, r = workspace
    @unpack xsub, lowsub, uppsub = workspace

    raa0 = 1e-5
    albefa = 0.1
    copyto!(r, fval)
    fill!(xsub, 1.0)

    for j in 1:n
        if iter < 2.5
            low[j] = xval[j] - asyinit * (xmax[j] - xmin[j])
            upp[j] = xval[j] + asyinit * (xmax[j] - xmin[j])
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

            low[j] = max(xval[j] - 10 * (xmax[j] - xmin[j]), min(low[j], xval[j] - 0.01 * (xmax[j] - xmin[j])))
            upp[j] = max(xval[j] + 0.01 * (xmax[j] - xmin[j]), min(upp[j], xval[j] + 10 * (xmax[j] - xmin[j])))
        end

        alfa[j] = max(xmin[j], low[j] + albefa * (xval[j] - low[j]), xval[j] - move * (xmax[j] - xmin[j]))
        beta[j] = min(xmax[j], upp[j] - albefa * (upp[j] - xval[j]), xval[j] + move * (xmax[j] - xmin[j]))

        xsub[j] = 0.5 * (alfa[j] + beta[j])
        lowsub[j] = alfa[j]
        uppsub[j] = beta[j]

        df_plus = max(df0dx[j], 0)
        df_minus = max(-df0dx[j], 0)

        p0[j] = (upp[j] - xval[j])^2 * (1.001 * df_plus + 0.001 * df_minus + raa0 / (xmax[j] - xmin[j]))
        q0[j] = (xval[j] - low[j])^2 * (0.001 * df_plus + 1.001 * df_minus + raa0 / (xmax[j] - xmin[j]))
        for i in 1:m
            dg_plus = max(dfdx[i, j], 0)
            dg_minus = max(-dfdx[i, j], 0)

            p[i, j] = (upp[j] - xval[j])^2 * (1.001 * dg_plus + 0.001 * dg_minus + raa0 / (xmax[j] - xmin[j]))
            q[i, j] = (xval[j] - low[j])^2 * (0.001 * dg_plus + 1.001 * dg_minus + raa0 / (xmax[j] - xmin[j]))
            r[i] -= p[i, j] / (upp[j] - xval[j]) + q[i, j] / (xval[j] - low[j])
        end
    end

    function obj(in::Vector, grad::Vector)
        x = in[1:n]
        y = in[n+1:n+m]
        z = in[end]

        f0 = 0.0
        # x
        for j in 1:n
            f0 += p0[j] / (upp[j] - x[j]) + q0[j] / (x[j] - low[j])
            grad[j] = p0[j] / (upp[j] - x[j])^2 - q0[j] / (x[j] - low[j])^2
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
    function cons(in::Vector, grad::Vector, i)
        x = in[1:n]
        y = in[n+1:n+m]
        z = in[end]

        f = r[i]
        # x
        for j in 1:n
            f += p[i, j] / (upp[j] - x[j]) + q[i, j] / (x[j] - low[j])
            grad[j] = p[i, j] / (upp[j] - x[j])^2 - q[i, j] / (x[j] - low[j])^2
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

    opt = NLopt.Opt(:LD_CCSAQ, n + m + 1)
    NLopt.xtol_rel!(opt, 1e-4)
    NLopt.maxtime!(opt, 1.0) # NOTE: hardcoded limit of 1s for the subproblem, might need to be increased for large problems
    NLopt.lower_bounds!(opt, lowsub)
    NLopt.upper_bounds!(opt, uppsub)
    NLopt.min_objective!(opt, obj)
    for i in 1:m
        NLopt.inequality_constraint!(opt, (x, g) -> cons(x, g, i))
    end

    NLopt.optimize!(opt, xsub)
    return @view(xsub[1:n])
end
