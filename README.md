# TopOpt Tools in Julia
This package provides tools for Finite Element Analysis, variable updating via the Method of Moving Asymptotes and filtering.
They implement an abstraction layer to functionality commonly used in density-based topology optimization scripts.
Check the exported members of module [`TopOptTools`](src/TopOptTools.jl) for an overview of the available tools.

Finite element analyses use [`Ferrite.jl`](https://github.com/Ferrite-FEM/Ferrite.jl) finite element toolbox.
Derivatives of the stiffness matrix with respect to the element variables are computed via automatic differentiation - [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl).
Linear systems can be solved using MKL Pardiso's sparse direct solver via [`Pardiso.jl`](https://github.com/JuliaSparse/Pardiso.jl) or Algebraic Multigrid Preconditioned Conjugate Gradient (AMG-PCG) sparse iterative solver from [`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl) and [`AlgebraicMultigrid.jl`](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl).

Parallelization is done via multithreading, which enables some level of scalability on the problem size without many changes in the structure of the program and of the data.

MMA subproblems are solved with [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). The dual problem is solved with the default configuration for box constrained optimizations: Fminbox (a barrier method) with L-BGFS as line search algorithm.

> [!NOTE]
> The `TopOptTools` module uses [`MKL.jl`](https://github.com/JuliaLinearAlgebra/MKL.jl) and [`MKLSparse.jl`](https://github.com/JuliaSparse/MKLSparse.jl) instead of the default BLAS backend. This may affect multithreaded code that interacts with it

## Installation
To add this package as a dependency to your project, run `]add https://gitlab.tudelft.nl/shapingmatterlab/TopOptTools.jl.git`.

To run the examples, start Julia from the root directory with:
```console
// sync environment from examples/Manifest.toml
$ julia --project=examples/ -e "using Pkg; Pkg.instantiate()"

$ julia --project=examples/ -t auto
julia> include("examples/1-simp.jl")
julia> simp(0.5, 0.5)
```

Some reasonable parameters that can be tested in each example:
1. `simp(0.5, 0.5)`
2. `bracket(0.2, 5.0, 7.0, filename="path/to/results")`
3. `thermal_actuator(0.3, 0.5)`
4. `mm_ecotopopt(350, 0.5, 0.5, 0.5)`

> [!TIP]
> If `filename` is set to `"results/bracket"`, the script will create the files:
> - `results/bracket.xxxx.vtu`: VTK Unstructured Grid File for each iteration
> - `results/bracket.pvd`: Paraview Data File to visualize all iterations as a time series
> - `results/bracket.h5`: HDF5 file containing the convergence history. Data is flattened to be compatible with other HDF5 readers
> - `results/bracket.design.jld2`: JLD2 file containing the design variables at the final iteration. It is HDF5-compatible but best suited for post-processing by Julia code
>
> The full path to the parent directory (`results/` in this case) must already exist, and existing files in it may be overwritten

## Structure
In this repository you can find:
- `src/`: source code for the module `TopOptTools`
- `examples/`: collection of scripts showing the usage of the package for different optimization formulations. The directory contains its own `Project.toml` and `Manifest.toml` defining an environment to run the scripts
  - `examples/models/`: `.geo` and `.msh` [Gmsh](https://gmsh.info/) files. The meshes are imported to the scripts using [`FerriteGmsh.jl`](https://github.com/Ferrite-FEM/FerriteGmsh.jl)

## Author(s)
This software has been developed by 
**Gustavo Asai** ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0009-0007-6133-6471](https://orcid.org/0009-0007-6133-6471), Technische Universiteit Delft

## License

The contents of this repository are licensed under an **Apache License v2.0** (see [LICENSE](LICENSE))

Copyright notice:

Technische Universiteit Delft hereby disclaims all copyright interest in the program "TopOptTools.jl". It is a Julia package with tools for density-based topology optimization scripts.
Henri Werij, Faculty of Aerospace Engineering, Technische Universiteit Delft.

&copy; 2026, G.J. Asai
