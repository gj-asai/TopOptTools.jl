module TopOpt

using MKL,
    Ferrite,
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

export Material, Isotropic2D, Orthotropic2D, Orthotropic3D,
    NodalLoad, LinearLoad,
    MaterialInterpolation, FEModel, get_dim, get_nvar,
    DesignVector, MMA,
    FEResults, fea!, stress,
    ConvolutionFilter, PDEFilter, filter!

const timer = TimerOutput()

include("design_variables.jl")

# Finite element analysis
include("material.jl")
include("loads.jl")
include("femodel.jl")
include("fea.jl")

# Optimization
include("MMA/MMA.jl")
include("filter_convolution.jl")
include("filter_pde.jl")

end # module TopOpt
