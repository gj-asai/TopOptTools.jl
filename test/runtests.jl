using Test
using TopOpt

using Ferrite
using Tensors

@testset "TopOpt.jl" begin
    include("material.jl")
    include("fea.jl")
end
nothing
