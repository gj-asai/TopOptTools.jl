# TopOpt Tools in Julia
This package provides tools for Finite Element Analysis, variable updating via the Method of Moving Asymptotes and filtering.
They implement an abstraction layer to functionality commonly used in density-based topology optimization scripts.
Check the exported members of module `TopOptTools` for an overview of the available tools.

Mechanical (linear elasticity) analysis and PDE filter use [`Ferrite.jl`](https://github.com/Ferrite-FEM/Ferrite.jl) finite element toolbox.
Derivatives of the stiffness matrix with respect to the element variables are computed via automatic differentiation - [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl).
The linear systems are solved using MKL Pardiso's sparse direct solver via [`Pardiso.jl`](https://github.com/JuliaSparse/Pardiso.jl), reusing symbolic and numerical factorizations when possible.

Parallelization is done via multithreading, which enables some level of scalability on the problem size without many changes in the structure of the program and of the data.
Task scheduling during multithreaded finite element matrices assembly is handled using [`OhMyThreads.jl`](https://github.com/JuliaFolds2/OhMyThreads.jl) and uses all threads started with Julia.
The linear solver is kept at its default configuration, which uses half of the available threads.

MMA subproblems are solved with [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). The dual problem is solved with the default configuration for box constrained optimizations: Fminbox (a barrier method) with L-BGFS as line search algorithm.

> [!NOTE]
> The `TopOptTools` module uses [`MKL.jl`](https://github.com/JuliaLinearAlgebra/MKL.jl) instead of the default BLAS backend. This may affect multithreaded code that interacts with it

## Installation
To add this package as a dependency to your project, run `]add https://gitlab.tudelft.nl/shapingmatterlab/TopOptTools.jl.git`.

To run the examples, start Julia from the root directory with `julia --project=examples/ -t auto` and `include` the file of the example you want to run. Some reasonable parameters that can be tested in each example:
1. `simp(0.5, 0.5)`
2. `bracket(0.2, 5.0, 7.0, filename="path/to/results")`
3. `thermal_actuator(0.3, 0.5)`
4. `mm_ecotopopt(350, 0.5, 0.5, 0.5)`

> [!TIP]
> If `filename` is set to `"results/mbb"`, the script will create the files:
> - `results/mbb.xxxx.vtu`: VTK Unstructured Grid File for each iteration
> - `results/mbb.pvd`: Paraview Data File to visualize all iterations as a time series
> - `results/mbb.h5`: HDF5 file containing the convergence history. Data is flattened to be compatible with other HDF5 readers
> - `results/mbb.design.jld2`: JLD2 file containing the design variables at the final iteration. It is HDF5-compatible but best suited for post-processing by Julia code
>
> The full path to the parent directory (`results/` in this case) must already exist, and existing files in it may be overwritten

## Structure
In this repository you can find:
- `src/`: source code for the module `TopOptTools`
- `examples/`: collection of scripts showing the usage of the package for different optimization formulations. The directory contains its own `Project.toml` defining an environment to run the scripts
  - `examples/models/`: `.geo` and `.msh` [Gmsh](https://gmsh.info/) files. The meshes are imported to the scripts using [`FerriteGmsh.jl`](https://github.com/Ferrite-FEM/FerriteGmsh.jl)

## Author(s)
This software has been developed by 
**Gustavo Asai** ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0009-0007-6133-6471](https://orcid.org/0009-0007-6133-6471), Technische Universiteit Delft

## License

The contents of this repository are licensed under an **Apache License v2.0** (see [LICENSE](LICENSE))

Copyright notice:

Technische Universiteit Delft hereby disclaims all copyright interest in the program "TopOptTools.jl". It is a Julia package with tools for density-based topology optimization scripts.
Henri Werij, Faculty of Aerospace Engineering, Technische Universiteit Delft.

&copy; 2025, G.J. Asai
