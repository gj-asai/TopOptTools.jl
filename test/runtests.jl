using Test
using TopOpt

@testset "TopOpt.jl" begin
    include("material.jl")
    include("fea.jl")
    include("filters.jl")
    include("mma.jl")
end

nothing
