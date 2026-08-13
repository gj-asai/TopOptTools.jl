"""
Contains the data necessary for the linear system solve and stores the results. Different physics are implementations of `FEA`.
`nvar` is the number of design variables per element.
"""
abstract type FEA{nvar} end

"""
Contains the task local data for parallel stiffness assemble
"""
struct StiffnessScratch{CC<:CellCache,CV<:CellValues,JCFG<:ForwardDiff.JacobianConfig,A<:Ferrite.AbstractAssembler}
    cell_cache::CC
    cellvalues::CV
    Ke::Matrix{Float64}
    jac::Matrix{Float64}
    cfg::JCFG
    assembler::A
end
function StiffnessScratch(model::FEModel{dim}, asm::Ferrite.AbstractAssembler, nvar::Int) where {dim}
    cell_cache = CellCache(model.dh)
    cellvalues = copy(model.cellvalues)
    n_basefuncs = getnbasefunctions(cellvalues)

    Ke = zeros(n_basefuncs, n_basefuncs)
    jac = zeros(n_basefuncs * n_basefuncs, nvar)
    cfg = ForwardDiff.JacobianConfig(
        (Ke, xe) -> element_stiffness!(Ke, xe, cell_cache, model),
        Ke, zeros(nvar),
        ForwardDiff.Chunk{nvar}()
    )

    return StiffnessScratch(cell_cache, cellvalues, Ke, jac, cfg, asm)
end

"""
    solve!(solver::FEA, f::AbstractVector)

Solves the finite element problem with the stiffness matrix stored in `solver` and right hand side `f`.
Stores the result in `solver.solution`
"""
function solve!(solver::FEA, f::AbstractVector)
    @unpack K, solution, ps = solver

    # solve linear system and store result in solver.solution
    pardiso(ps, solution, tril(K), f)
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE) # reuse K for next solves
end

"""
    adjoint_sensitivities!(dJdx, lambda, displacements, solver::FEA)

Uses the adjoint system to compute the sensitivities ∂J/∂x of a function J(u).

`lambda` is the vector of adjoint variables, obtained from calling `fea!` with right hand side dJ/du.
`displacements` is the vector of nodal displacements, obtained from calling `fea!` with right hand side equal to the forces vector
"""
function adjoint_sensitivities!(dJdx, lambda, displacements, solver::FEA{nvar}) where {nvar}
    model = solver.model
    ∂Ke∂x = solver.∂Ke∂x
    for cell in CellIterator(model.dh)
        e = cellid(cell)
        dofs = celldofs(cell)

        @views ue, lambdae = lambda[dofs], displacements[dofs]
        for i in 1:nvar
            dJdx[nvar*(e-1)+i] = -dot(lambdae, ∂Ke∂x[nvar*(e-1)+i], ue)
        end
    end
end

"""
    update_stiffness!(solver::FEA, x::AbstractVector)

Recomputes the stiffness matrix, using the new design variables `x`.
Needed to update the stiffness matrix before calling `fea!` and `adjoint_sensitivities!`
"""
function update_stiffness!(solver::FEA{nvar}, x::AbstractVector) where {nvar}
    solver.x .= x

    @unpack ps, x, K, ∂Ke∂x, chnl = solver
    model = solver.model

    n_basefuncs = getnbasefunctions(model.cellvalues)

    start_assemble(K) # zero K out before starting
    Threads.@threads for e in 1:getncells(model.grid)
        scratch = take!(chnl)
        @unpack cell_cache, cellvalues, Ke, jac, cfg, assembler = scratch

        Ferrite.reinit!(cell_cache, e)
        Ferrite.reinit!(cellvalues, cell_cache)

        xe = @view x[nvar*(e-1)+1:nvar*e]

        # update Ke and obtain ∂Ke∂x via automatic differentiation
        ForwardDiff.jacobian!(
            jac,
            (Ke, xe) -> element_stiffness!(Ke, xe, cellvalues, solver),
            Ke, xe, cfg,
            Val{false}() # disable tag checking because we are rebuilding the anonymous function
        )
        for var_idx = 1:nvar
            @views ∂Ke∂x[nvar*(e-1)+var_idx] = reshape(jac[:, var_idx], n_basefuncs, n_basefuncs)
        end

        assemble!(assembler, celldofs(cell_cache), Ke)
        put!(chnl, scratch)
    end

    # apply boundary conditions
    apply!(K, model.ch)

    # tell Pardiso to recalculate the numerical factorization in the next solve
    # the sparsity pattern of K doesnt change, skip the symbolic factorization
    if get_phase(ps) == Pardiso.SOLVE_ITERATIVE_REFINE
        set_phase!(ps, Pardiso.NUM_FACT_SOLVE_REFINE)
    end
end

# xe is a view of the design vector that contains the variables corresponding to one element
function element_stiffness!(::Matrix, ::AbstractVector, ::CellValues, solver::FEA)
    throw("Method element_stiffness!(Ke::Matrix, xe::AbstractVector, cv::CellValues, ::$(typeof(solver))) is not defined")
end
