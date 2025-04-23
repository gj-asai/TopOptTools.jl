# Implementation following
# https://people.kth.se/~krille/mmagcmma.pdf
module MMA

using UnPack,
    LinearAlgebra,
    TimerOutputs

using ..TopOpt

export Objective, Constraints,
    optimize,
    relative_change

include("function_containers.jl")
include("mma_problem.jl")
include("convex_approximation.jl")
include("subproblem.jl")
include("optimize.jl")

end
