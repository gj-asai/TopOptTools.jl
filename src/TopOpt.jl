module TopOpt

using MKL, Pardiso,
    NLopt,
    Ferrite,
    Tensors,
    LinearAlgebra,
    SparseArrays,
    ForwardDiff,
    NearestNeighbors,
    OhMyThreads,
    UnPack,
    TimerOutputs

export Material, Isotropic2D, Isotropic3D, Orthotropic2D, Orthotropic3D,
    NodalLoad, LinearLoad,
    MaterialInterpolation, FEModel, get_dim, get_nvar,
    DesignVector, MMAWorkspace, mma_update!,
    FESolver, update_xPhys!, fea!,
    ConvolutionFilter, PDEFilter, filter!,
    project_heaviside!, project_heaviside_derivative!

const timer = TimerOutput()

include("design_variables.jl")

# Finite element analysis
include("material.jl")
include("loads.jl")
include("femodel.jl")
include("fea.jl")

# MMA
include("mma.jl")

# Filtering
include("filter_convolution.jl")
include("filter_pde.jl")
include("projection.jl")

end # module TopOpt
