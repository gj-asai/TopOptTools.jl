module TopOpt

using MKL
using Pardiso,
    Optim,
    Ferrite,
    Tensors,
    LinearAlgebra,
    SparseArrays,
    ForwardDiff,
    NearestNeighbors,
    OhMyThreads,
    UnPack

export Material, Isotropic2D, Isotropic3D, Orthotropic2D, Orthotropic3D,
    NodalLoad, LinearLoad,
    MaterialInterpolation, FEModel,
    DesignVector, MMAWorkspace, mma_update!,
    FESolver, fea!, update_stiffness!, compute_force_vector,
    ConvolutionFilter, PDEFilter, filter!,
    project_heaviside!, project_heaviside_derivative!

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
