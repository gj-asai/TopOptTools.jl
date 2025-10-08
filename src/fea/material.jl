"Holds the elastic tensor of the material and optionally its mass density and CO2 intensity"
struct Material{dim,T<:Real,CT<:SymmetricTensor}
    C::CT
    ρ::T
    CO2::T
end

"""
    Isotropic2D(; E, nu[, ρ, CO2])

Creates a `Material` instance from 2D isotropic properties
"""
function Isotropic2D(; E, nu, ρ=0.0, CO2=0.0)
    S = [1/E -nu/E 0;
        -nu/E 1/E 0;
        0 0 2(1+nu)/E]
    C = fromvoigt(SymmetricTensor{4,2,Float64}, inv(S))
    return Material{2,Float64,typeof(C)}(C, ρ, CO2)
end

"""
    Isotropic3D(; E, nu[, ρ, CO2])

Creates a `Material` instance from 3D isotropic properties
"""
function Isotropic3D(; E, nu, ρ=0.0, CO2=0.0)
    S = [1/E -nu/E -nu/E 0 0 0;
        -nu/E 1/E -nu/E 0 0 0;
        -nu/E -nu/E 1/E 0 0 0;
        0 0 0 2(1+nu)/E 0 0;
        0 0 0 0 2(1+nu)/E 0;
        0 0 0 0 0 2(1+nu)/E]
    C = fromvoigt(SymmetricTensor{4,3,Float64}, inv(S))
    return Material{3,Float64,typeof(C)}(C, ρ, CO2)
end

"""
    Orthotropic2D(; El, Et, nult, Glt[, ρ, CO2])

Creates a `Material` instance from 2D isotropic properties
"""
function Orthotropic2D(; El, Et, nult, Glt, ρ=0.0, CO2=0.0)
    S = [1/El -nult/El 0;
        -nult/El 1/Et 0;
        0 0 1/Glt]
    C = fromvoigt(SymmetricTensor{4,2,Float64}, inv(S))
    return Material{2,Float64,typeof(C)}(C, ρ, CO2)
end

"""
    Orthotropic3D(; El, Et, nult, Glt[, ρ, CO2])

Creates a `Material` instance from 3D isotropic properties.

The transversal Poisson's ratio is calculated following:
R. Christensen
Tensor Transformations and Failure Criteria for the Analysis of Fiber Composite Materials
Journal of Composite Materials, 1988
"""
function Orthotropic3D(; El, Et, nult, Glt, ρ=0.0, CO2=0.0)
    nutt = nult * (1 - nult * El / Et) / (1 - nult)
    S = [1/El -nult/El -nult/El 0 0 0;
        -nult/El 1/Et -nutt/Et 0 0 0;
        -nult/El -nutt/Et 1/Et 0 0 0;
        0 0 0 2(1+nutt)/Et 0 0;
        0 0 0 0 1/Glt 0;
        0 0 0 0 0 1/Glt]
    C = fromvoigt(SymmetricTensor{4,3,Float64}, inv(S))
    return Material{3,Float64,typeof(C)}(C, ρ, CO2)
end

# void materials in 2 and 2 dimensions
const void2D = Isotropic2D(E=1e-3, nu=0.3)
const void3D = Isotropic3D(E=1e-3, nu=0.3)
void(::Val{2}) = void2D
void(::Val{3}) = void3D
