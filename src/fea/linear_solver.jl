abstract type LinearSolver end

struct DirectSolver <: LinearSolver
    ps::MKLPardisoSolver
end
function DirectSolver()
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    set_iparm!(ps, 1, 1) # to be able to manually set iparm
    set_iparm!(ps, 12, 1) # tells Pardiso we are giving a CSC matrix instead of CSR
    return DirectSolver(ps)
end

function update_matrix!(solver::DirectSolver, ::AbstractMatrix)
    # tell Pardiso to recalculate the numerical factorization in the next solve
    # the sparsity pattern of K doesnt change, skip the symbolic factorization
    if get_phase(solver.ps) == Pardiso.SOLVE_ITERATIVE_REFINE
        set_phase!(solver.ps, Pardiso.NUM_FACT_SOLVE_REFINE)
    end
end

function solve!(solver::DirectSolver, solution::AbstractVector, K::AbstractMatrix, f::AbstractVector)
    pardiso(solver.ps, solution, tril(K), f)
    set_phase!(solver.ps, Pardiso.SOLVE_ITERATIVE_REFINE) # reuse K for next solves
end
