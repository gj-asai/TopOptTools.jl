# Implementation following
# https://people.kth.se/~krille/mmagcmma.pdf
module MMA

using UnPack,
    LinearAlgebra,
    Statistics,
    OhMyThreads,
    TimerOutputs

using ..TopOpt

export Objective, Constraints,
    MMAProblem, iterate,
    relative_change

include("function_containers.jl")
include("artificial.jl")
include("mma_state.jl")
include("convex_approximation.jl")
include("subproblem.jl")
include("mma_problem.jl")

end
