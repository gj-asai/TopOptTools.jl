module TopOptTools

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

export Material, Isotropic2D, Isotropic3D, Orthotropic2D, Orthotropic3D
export MaterialInterpolation
export NodalLoad, LinearLoad
export FEModel
export FESolver, fea!, adjoint_sensitivities!, update_stiffness!, compute_force_vector
export MMAWorkspace, mma_update!
export ConvolutionFilter, PDEFilter, filter!
export project_heaviside!, project_heaviside_derivative!

# Finite element analysis
include("fea/material.jl")
include("fea/material_interpolation.jl")
include("fea/loads.jl")
include("fea/femodel.jl")
include("fea/fea.jl")

# MMA
include("mma/mma.jl")

# Filtering and projection
include("utils/filter_convolution.jl")
include("utils/filter_pde.jl")
include("utils/projection.jl")

end # module TopOptTools
