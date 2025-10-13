# TopOpt tools in Julia
This package provides tools for Finite Element Analysis, variable updating via the Method of Moving Asymptotes and filtering.
They implement an abstraction layer to functionality commonly used in density-based topology optimization scripts.

Parallelization is done via multithreading, which enables some level of scalability on the problem size without many changes in the structure of the program and of the data.
It uses all threads created when Julia started.

## Installation
To add this package as a dependency to your project, run `]add https://github.com/gj-asai/TopOpt.git`.

## Structure
In this repository you can find:
- `src/`: source code for the module `TopOpt`
- `examples/`: collection of scripts showing the usage of the package for different optimization formulations. The directory contains its own `Project.toml` defining an environment to run the scripts.

## Author(s)
This software has been developed by 
**Gustavo Asai** ![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png) [0009-0007-6133-6471](https://orcid.org/0009-0007-6133-6471), Technische Universiteit Delft

WIP
