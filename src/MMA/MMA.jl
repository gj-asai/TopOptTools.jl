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

@kwdef struct OptimOpts
    maxiter::Int = 200
    reltol::Float64 = 1e-3
    move::Float64 = 0.5
    asyinit::Float64 = 0.5
    asydecr::Float64 = 0.7
    asyincr::Float64 = 1.2
end

const timer = TimerOutput()

include("function_containers.jl")
include("mma_problem.jl")
include("convex_approximation.jl")
include("subproblem.jl")
include("optimize.jl")

end
