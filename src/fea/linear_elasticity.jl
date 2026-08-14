struct LinearElasticity{nvar,FEM<:FEModel,LS<:LinearSolver,MatInterp<:MaterialInterpolation{nvar},SS<:StiffnessScratch} <: FEA{nvar}
    model::FEM
    solver::LS

    x::Vector{Float64}
    mat_interp::MatInterp

    K::SparseMatrixCSC{Float64,Int}
    ∂Ke∂x::Vector{Matrix{Float64}}
    solution::Vector{Float64}
    chnl::Channel{SS}
end
function LinearElasticity(model::FEModel, mat_interp::MaterialInterpolation{nvar}; solver_type::Symbol=:direct) where {nvar}
    if solver_type == :direct
        solver = DirectSolver()
    elseif solver_type == :iterative
        solver = IterativeSolver(ndofs(model.dh))
    else
        throw("Invalid solver_type $(solver_type), must be :direct or :iterative")
    end

    # preallocate vectors
    x0 = Vector{Float64}(undef, getncells(model.grid) * nvar)
    solution = similar(x0, ndofs(model.dh))

    # preallocate stiffness and sensitivities
    n_basefuncs = getnbasefunctions(model.cellvalues)
    K = allocate_matrix(model.dh)
    ∂Ke∂x = fill(zeros(n_basefuncs, n_basefuncs), length(x0))

    # preallocate thread local containers
    chnl = Channel{StiffnessScratch}(Threads.nthreads())
    foreach(1:Threads.nthreads()) do _
        asm = start_assemble(K; fillzero=false, atomic=true)
        put!(chnl, StiffnessScratch(model, asm, nvar))
    end

    return LinearElasticity(model, solver, x0, mat_interp, K, ∂Ke∂x, solution, chnl)
end

function element_stiffness!(Ke::Matrix{T}, xe::AbstractVector, cellvalues::CellValues, fea::LinearElasticity) where {T<:Real}
    fill!(Ke, zero(T))

    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            ∇sδεi = shape_symmetric_gradient(cellvalues, q_point, i)
            for j in 1:i
                ∇sδεj = shape_symmetric_gradient(cellvalues, q_point, j)
                Ke[i, j] += ∇sδεi ⊡ interpolate(xe, fea.mat_interp) ⊡ ∇sδεj * dΩ
            end
        end
    end

    for i in axes(Ke, 1), j in axes(Ke, 1)[begin+i:end]
        Ke[i, j] = Ke[j, i]
    end
end
