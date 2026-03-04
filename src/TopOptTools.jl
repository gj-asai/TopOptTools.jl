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
export FEModel
export FESolver, fea!, adjoint_sensitivities!, update_stiffness!
export NodalLoad, LinearLoad, EigenstrainLoad, compute_rhs, compute_rhs!
export MMAWorkspace, mma_update!, restart!
export ConvolutionFilter, PDEFilter, filter!
export project_heaviside!, project_heaviside_derivative!

# Finite element analysis
include("fea/material.jl")
include("fea/material_interpolation.jl")
include("fea/femodel.jl")
include("fea/fea.jl")
include("fea/loads.jl")

# MMA
include("mma/mma.jl")

# Filtering and projection
include("filtering/convolution.jl")
include("filtering/pde.jl")
include("filtering/heaviside.jl")

end # module TopOptTools
