using Test
using TopOpt

using Ferrite
using Tensors

@testset "TopOpt.jl" begin
    include("material.jl")
    include("fea.jl")
end

@testset "MMA.jl" begin
    include("mma.jl")
end

nothing
