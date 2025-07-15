mutable struct PDEFilter{TK<:SparseMatrixCSC,TT<:SparseMatrixCSC,SS<:MKLPardisoSolver}
    Kf::TK
    T::TT
    ps::SS
end

"""
    PDEFilter(radius::Float64, model::FEModel)

Creates an isotropic Helmholtz PDE filter of radius `radius` for the mesh stored in `model`
"""
function PDEFilter(radius::Float64, model::FEModel)
    @unpack ip, qr, grid = model

    r_filter = radius / (2 * sqrt(3))
    Kd = r_filter^2 * one(Tensor{2,2}) # isotropic filter

    cellvalues = CellValues(qr, ip)
    dh = DofHandler(grid)
    add!(dh, :x, ip)
    close!(dh)

    Kf = allocate_matrix(dh)
    T = spzeros(ndofs(dh), getncells(grid))

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(Kf)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues, cell)
        fill!(Ke, 0.0)

        @inbounds for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                Ni = shape_value(cellvalues, q_point, i)
                ∇Ni = shape_gradient(cellvalues, q_point, i)
                T[celldofs(cell)[i], cellid(cell)] += Ni * dΩ
                for j in 1:i
                    Nj = shape_value(cellvalues, q_point, j)
                    ∇Nj = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇Ni ⋅ Kd ⋅ ∇Nj + Ni ⋅ Nj) * dΩ
                end
            end
        end

        for i in 1:size(Ke, 1), j in i+1:size(Ke, 1)
            Ke[i, j] = Ke[j, i]
        end

        assemble!(assembler, celldofs(cell), Ke)
    end
    tril!(Kf) # Pardiso solver will use only the lower part of the matrix

    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    set_iparm!(ps, 1, 1) # to be able to manually set iparm
    set_iparm!(ps, 12, 1) # tells Pardiso we are giving a CSC matrix instead of CSR

    f = PDEFilter(Kf, T, ps)
    finalizer(f) do x
        set_phase!(x.ps, Pardiso.RELEASE_ALL)
    end
end

"""
    filter!(x::AbstractVector, f::PDEFilter)

In-place filtering of `x`
"""
function filter!(x::AbstractVector, f::PDEFilter)
    x_filt = Vector{Float64}(undef, size(f.T, 1))
    pardiso(f.ps, x_filt, f.Kf, f.T * x)
    x .= f.T' * x_filt

    # Kf doesnt change, so next solves can reuse the factorization
    set_phase!(f.ps, Pardiso.SOLVE_ITERATIVE_REFINE)
end

"""
    filter!(x::AbstractVector, weight::AbstractVector, f::PDEFilter)

In-place filtering of `x` using weights `weight`:
weight * xfiltered = H * (weight * x)
"""
function filter!(x::AbstractVector, weight::AbstractVector, f::PDEFilter)
    x .*= weight
    filter!(x, f)
    x ./= weight
end
