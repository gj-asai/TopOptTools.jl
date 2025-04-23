module TopOpt

using MKL
using Ferrite,
    Tensors,
    LinearAlgebra,
    SparseArrays,
    Sparspak,
    LinearSolve,
    ForwardDiff,
    NearestNeighbors,
    OhMyThreads,
    UnPack,
    TimerOutputs

export Isotropic2D, Orthotropic2D, Orthotropic3D,
    SIMP, SOMP, MMSOMP,
    NodalLoad, LinearLoad,
    FEModel,
    DesignVariables, MMA,
    FEResults, fea!, stress,
    ConvolutionFilter, filter!,
    compliance, dcompliance,
    impact, dimpact,
    volume, dvolume

const timer = TimerOutput()

include("design_variables.jl")

# Finite element analysis
include("material.jl")
include("loads.jl")
include("femodel.jl")
include("fea.jl")

# Optimization
include("MMA/MMA.jl")
include("filters.jl")
include("interpolations/simp.jl")
include("interpolations/somp.jl")
include("interpolations/mmsomp.jl")

end # module TopOpt
