struct PDEFilter{TK<:SparseMatrixCSC,TT<:SparseMatrixCSC}
    Kf::TK
    T::TT
end

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

    return PDEFilter(Kf, T)
end

function filter!(x::AbstractVector, f::PDEFilter)
    x .= f.T' * LinearSolve.solve(
        LinearSolve.LinearProblem(f.Kf, f.T * x),
        KrylovJL_CG(),
    ).u
end

function filter!(x::AbstractVector, weight::AbstractVector, f::PDEFilter)
    x .*= weight
    filter!(x, f)
    x ./= weight
end
