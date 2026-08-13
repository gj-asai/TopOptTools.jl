struct LinearElasticity{FEM<:FEModel,SS<:StiffnessScratch} <: FEA
    model::FEM
    ps::MKLPardisoSolver

    x::Vector{Float64}
    K::SparseMatrixCSC{Float64,Int}
    ∂Ke∂x::Vector{Matrix{Float64}}
    solution::Vector{Float64}
    chnl::Channel{SS}
end
function LinearElasticity(model::FEModel)
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    set_iparm!(ps, 1, 1) # to be able to manually set iparm
    set_iparm!(ps, 12, 1) # tells Pardiso we are giving a CSC matrix instead of CSR

    # preallocate vectors
    x0 = Vector{Float64}(undef, getncells(model.grid) * get_nvar(model))
    solution = similar(x0, ndofs(model.dh))

    # preallocate stiffness and sensitivities
    n_basefuncs = getnbasefunctions(model.cellvalues)
    K = allocate_matrix(model.dh)
    ∂Ke∂x = fill(zeros(n_basefuncs, n_basefuncs), length(x0))

    # preallocate thread local containers
    chnl = Channel{StiffnessScratch}(Threads.nthreads())
    foreach(1:Threads.nthreads()) do _
        asm = start_assemble(K; fillzero=false, atomic=true)
        put!(chnl, StiffnessScratch(model, asm))
    end

    return LinearElasticity(model, ps, x0, K, ∂Ke∂x, solution, chnl)
end

function element_stiffness!(Ke::Matrix{T}, xe::AbstractVector, cellvalues::CellValues, solver::LinearElasticity) where {T<:Real}
    fill!(Ke, zero(T))

    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            ∇sδεi = shape_symmetric_gradient(cellvalues, q_point, i)
            for j in 1:i
                ∇sδεj = shape_symmetric_gradient(cellvalues, q_point, j)
                Ke[i, j] += ∇sδεi ⊡ interpolate(xe, solver.model.mat_interp) ⊡ ∇sδεj * dΩ
            end
        end
    end

    for i in axes(Ke, 1), j in axes(Ke, 1)[begin+i:end]
        Ke[i, j] = Ke[j, i]
    end
end
