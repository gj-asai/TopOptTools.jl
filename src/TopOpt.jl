module TopOpt

using Ferrite,
    Tensors,
    SparseArrays,
    Krylov,
    ForwardDiff,
    Nonconvex,
    NonconvexMMA,
    NearestNeighbors,
    OhMyThreads,
    UnPack,
    TimerOutputs

Nonconvex.@load MMA

export Isotropic2D, Orthotropic2D, Orthotropic3D,
    SIMP, SOMP, MMSOMP,
    NodalLoad, LinearLoad,
    FEModel,
    FEResults, fea!,
    ConvolutionFilter, filter!,
    OptimOpts, topopt,
    compliance, dcompliance,
    impact, dimpact,
    volume, dvolume

# Finite element analysis
include("material.jl")
include("loads.jl")
include("femodel.jl")
include("fea.jl")

# Optimization
include("filters.jl")
include("optimization.jl")
include("interpolations/simp.jl")
include("interpolations/somp.jl")
include("interpolations/mmsomp.jl")

end # module TopOpt
