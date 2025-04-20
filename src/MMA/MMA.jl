# Implementation following
# https://people.kth.se/~krille/mmagcmma.pdf
module MMA

using UnPack,
    LinearAlgebra,
    TimerOutputs

using ..TopOpt

export Objective, Constraints
export optimize

include("function_containers.jl")
include("mma_problem.jl")
include("convex_approximation.jl")
include("solve_primal_dual.jl")
include("optimize.jl")

end
