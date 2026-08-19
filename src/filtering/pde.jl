# task-local scratch data to build filter in parallel
struct PDEFilterScratch{CC<:CellCache,CV<:CellValues,A<:Ferrite.AbstractAssembler}
    cell_cache::CC
    cellvalues::CV

    # preallocated element arrays
    Ke::Matrix{Float64}
    Te::Vector{Float64}
    TXe::Vector{Float64}

    # thread-local interpolation sparse matrices
    I_::Vector{Int}
    J_::Vector{Int}
    T_::Vector{Float64}
    TX_::Vector{Float64}

    # for stiffness matrix
    assembler::A
end

function PDEFilterScratch(cv::CellValues, dh::DofHandler, asm::Ferrite.AbstractAssembler)
    cell_cache = CellCache(dh)
    cellvalues = copy(cv)

    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(n_basefuncs, n_basefuncs)
    Te = zeros(n_basefuncs)
    TXe = zeros(n_basefuncs)

    I_, J_ = Int[], Int[]
    T_, TX_ = Float64[], Float64[]

    return PDEFilterScratch(cell_cache, cellvalues, Ke, Te, TXe, I_, J_, T_, TX_, asm)
end

struct PDEFilter{M<:SparseMatrixCSC}
    Kf::M # stiffness matrix
    T::M  # project elements to nodes
    TX::M # project nodes to elements
    ps::MKLPardisoSolver # linear solver
    xnodal::Vector{Float64}
    rhs::Vector{Float64}
end

"""
    PDEFilter(radius, model::FEModel)

Creates an isotropic Helmholtz PDE filter of radius `radius` for the mesh stored in `model`.
"""
function PDEFilter(radius, model::FEModel{dim}) where {dim}
    @unpack ip, qr, grid = model

    r_filter = radius / (2 * sqrt(3))
    Kd = r_filter^2 * one(Tensor{2,dim}) # isotropic filter

    cv = CellValues(qr, ip)
    dh = DofHandler(grid)
    add!(dh, :x, ip)
    close!(dh)

    # preallocate matrices
    Kf = allocate_matrix(dh)
    chnl = Channel{PDEFilterScratch}(Threads.nthreads())
    foreach(1:Threads.nthreads()) do _
        asm = start_assemble(Kf; fillzero=false)
        put!(chnl, PDEFilterScratch(cv, dh, asm))
    end

    n_basefuncs = getnbasefunctions(cv)
    start_assemble(Kf)
    for color in model.colors
        Threads.@threads for e in color
            scratch = take!(chnl)
            @unpack cell_cache, cellvalues, Ke, Te, TXe, I_, J_, T_, TX_, assembler = scratch

            Ferrite.reinit!(cell_cache, e)
            Ferrite.reinit!(cellvalues, cell_cache)

            fill!(Ke, 0.0)
            fill!(Te, 0.0)
            fill!(TXe, 0.0)

            for q_point in 1:getnquadpoints(cellvalues)
                dΩ = getdetJdV(cellvalues, q_point)
                for i in 1:n_basefuncs
                    Ni = shape_value(cellvalues, q_point, i)
                    ∇Ni = shape_gradient(cellvalues, q_point, i)
                    Te[i] += Ni * dΩ
                    TXe[i] += Ni * dΩ / model.elemvol[e]
                    for j in 1:i
                        Nj = shape_value(cellvalues, q_point, j)
                        ∇Nj = shape_gradient(cellvalues, q_point, j)
                        Ke[i, j] += (∇Ni ⋅ Kd ⋅ ∇Nj + Ni * Nj) * dΩ
                    end
                end
            end

            for i in axes(Ke, 1), j in axes(Ke, 1)[begin+i:end]
                Ke[i, j] = Ke[j, i]
            end

            append!(I_, celldofs(cell_cache))
            append!(J_, repeat([e], n_basefuncs))
            append!(T_, Te)
            append!(TX_, TXe)

            assemble!(assembler, celldofs(cell_cache), Ke)
            put!(chnl, scratch)
        end
    end
    close(chnl) # no more put!, now we can iterate on chnl
    tril!(Kf)

    # single-threaded final assembly
    I, J = Int[], Int[]
    VT, VTX = Float64[], Float64[]
    for scratch in chnl
        append!(I, scratch.I_)
        append!(J, scratch.J_)
        append!(VT, scratch.T_)
        append!(VTX, scratch.TX_)
    end

    T = sparse(I, J, VT, ndofs(dh), getncells(grid))
    TX = sparse(J, I, VTX, getncells(grid), ndofs(dh))
    xnodal = Vector{Float64}(undef, ndofs(dh))
    rhs = Vector{Float64}(undef, ndofs(dh))

    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    set_iparm!(ps, 1, 1) # to be able to manually set iparm
    set_iparm!(ps, 12, 1) # tells Pardiso we are giving a CSC matrix instead of CSR

    return PDEFilter(Kf, T, TX, ps, xnodal, rhs)
end

"""
    filter!(x::AbstractVector, f::PDEFilter)

In-place filtering of `x`
"""
function filter!(x::AbstractVector, f::PDEFilter)
    mul!(f.rhs, f.T, x) # f.rhs = f.T * x
    pardiso(f.ps, f.xnodal, f.Kf, f.rhs)
    mul!(x, f.TX, f.xnodal) # x = f.TX * f.xnodal

    set_phase!(f.ps, Pardiso.SOLVE_ITERATIVE_REFINE) # reuse factorization for next solves
end

"""
    filter!(x::AbstractVector, weight::AbstractVector, f::PDEFilter)

In-place filtering of `x` using weights `weight`:
weight * xfiltered = filter(weight * x)
"""
function filter!(x::AbstractVector, weight::AbstractVector, f::PDEFilter)
    x .*= weight
    filter!(x, f)
    x ./= weight
end
