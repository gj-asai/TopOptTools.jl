module TopOptTools

using MKL
using MKLSparse

using AlgebraicMultigrid
using Ferrite
using ForwardDiff
using Krylov
using LinearAlgebra
using NearestNeighbors
using Optim
using Pardiso
using SparseArrays
using Tensors
using UnPack

export Material, Isotropic2D, Isotropic3D, Orthotropic2D, Orthotropic3D
export MaterialInterpolation
export FEModel
export solve!, adjoint_sensitivities!, update_stiffness!
export LinearElasticity
export NodalLoad, LinearLoad, EigenstrainLoad, compute_rhs, compute_rhs!
export MMAWorkspace, mma_update!, restart!
export ConvolutionFilter, PDEFilter, filter!
export project_heaviside!, project_heaviside_derivative!

# Finite element analysis
include("fea/femodel.jl")
include("fea/linear_solver.jl")
include("fea/fea.jl")
include("fea/loads.jl")

include("fea/material.jl")
include("fea/material_interpolation.jl")
include("fea/linear_elasticity.jl")

# MMA
include("mma/mma.jl")

# Filtering and projection
include("filtering/convolution.jl")
include("filtering/pde.jl")
include("filtering/heaviside.jl")

end # module TopOptTools
