"""
Preallocates all memory used by the MMA solver
"""
struct MMAWorkspace
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
end

"""
    MMAWorkspace(m::Int, n::Int, a0, a, c, d; asyinit=0.5, asyincr=1.2, asydecr=0.7, move=0.5)

Prepares an optimization with `m` constraints and `n` design variables.

`a0`, `a`, `c` and `d` are MMA parameters
"""
function MMAWorkspace(m::Int, n::Int, a0, a, c, d; asyinit=0.5, asyincr=1.2, asydecr=0.7, move=0.5)
    MMAWorkspace(
        asyinit, asyincr, asydecr, move, a0, a, c, d, m, n,
        Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n),
        Vector{Float64}(undef, n), Vector{Float64}(undef, n),
        Matrix{Float64}(undef, m, n), Matrix{Float64}(undef, m, n), Vector{Float64}(undef, m),
    )
end

# TODO: change this signature
"""
    mma_update!(workspace, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx)

Returns the updated value of the design variables

The subproblem is solved using a dual method
"""
function mma_update!(workspace, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx)
    @unpack a0, a, c, d, asyinit, asyincr, asydecr, move, m, n = workspace
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

    function primal(lambda)
        plam = sqrt.(p0 + p' * lambda)
        qlam = sqrt.(q0 + q' * lambda)
        x = @. max(alfa, min((plam * low + qlam * upp) / (plam + qlam), beta))
        y = @. max(0, lambda - c)
        z = max(0, lambda ⋅ a - 1.0)
        return x, y, z
    end

    function dual_obj(lambda)
        x, y, z = primal(lambda)
        Lx = sum(p0[j] / (upp[j] - x[j]) + q0[j] / (x[j] - low[j]) for j in 1:n) +
             sum(lambda[i] * (sum((p[i, j] / (upp[j] - x[j]) + q[i, j] / (x[j] - low[j])) for j in 1:n) + r[i]) for i in 1:m)
        Ly = sum(c[i] * y[i] + y[i]^2 / 2 - lambda[i] * y[i] for i in 1:m)
        Lz = z + z^2 / 2 - lambda ⋅ a * z
        return -(Lx + Ly + Lz) # flipping the sign because we want to maximize the objective
    end

    function dual_jac!(J, lambda)
        x, y, z = primal(lambda)
        for i in 1:m
            J[i] = sum((p[i, j] / (upp[j] - x[j]) + q[i, j] / (x[j] - low[j])) for j in 1:n) + r[i] - a[i] * z - y[i]
        end
        J .*= -1.0 # because we flipped the sign of the objective
    end

    res = optimize(dual_obj, dual_jac!, zeros(m), fill(Inf, m), ones(m))
    lambda = Optim.minimizer(res)
    return primal(lambda)[1]
end
