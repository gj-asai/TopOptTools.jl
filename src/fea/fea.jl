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
function StiffnessScratch(model::FEModel, asm::Ferrite.AbstractAssembler, nvar::Int)
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
    solve!(fea::FEA, f::AbstractVector)

Solves the finite element problem with the stiffness matrix stored in `fea` and right hand side `f`.
Stores the result in `fea.solution`
"""
solve!(fea::FEA, f::AbstractVector) = solve!(fea.solver, fea.solution, fea.K, f)

"""
    adjoint_sensitivities!(dJdx, lambda, displacements, fea::FEA)

Uses the adjoint system to compute the sensitivities ∂J/∂x of a function J(u).

`lambda` is the vector of adjoint variables, obtained from calling `solve!` with right hand side dJ/du.
`displacements` is the vector of nodal displacements, obtained from calling `solve!` with right hand side equal to the forces vector
"""
function adjoint_sensitivities!(dJdx, lambda, displacements, fea::FEA{nvar}) where {nvar}
    model = fea.model
    ∂Ke∂x = fea.∂Ke∂x
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
    update_stiffness!(fea::FEA, x::AbstractVector)

Recomputes the stiffness matrix, using the new design variables `x`.
Needed to update the stiffness matrix before calling `solve!` and `adjoint_sensitivities!`
"""
function update_stiffness!(fea::FEA{nvar}, x::AbstractVector) where {nvar}
    fea.x .= x

    @unpack x, K, ∂Ke∂x, chnl = fea
    model = fea.model

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
            (Ke, xe) -> element_stiffness!(Ke, xe, cellvalues, fea),
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

    # tell the linear solver that the matrix changed
    update_matrix!(fea.solver, K)
end

# xe is a view of the design vector that contains the variables corresponding to one element
function element_stiffness!(::Matrix, ::AbstractVector, ::CellValues, fea::FEA)
    throw("Method element_stiffness!(Ke::Matrix, xe::AbstractVector, cv::CellValues, ::$(typeof(fea))) is not defined")
end
