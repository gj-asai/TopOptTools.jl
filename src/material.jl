struct Material{dim,T<:Real,CT<:SymmetricTensor}
    C::CT
    ρ::T
    CO2::T
end

function Isotropic2D(; E, nu, ρ=0.0, CO2=0.0)
    S = [1/E -nu/E 0;
        -nu/E 1/E 0;
        0 0 2(1+nu)/E]
    C = fromvoigt(SymmetricTensor{4,2,Float64}, inv(S))
    return Material{2,Float64,typeof(C)}(C, ρ, CO2)
end

function Orthotropic2D(; El, Et, nult, Glt, ρ=0.0, CO2=0.0)
    S = [1/El -nult/El 0;
        -nult/El 1/Et 0;
        0 0 1/Glt]
    C = fromvoigt(SymmetricTensor{4,2,Float64}, inv(S))
    return Material{2,Float64,typeof(C)}(C, ρ, CO2)
end

function Orthotropic3D(; El, Et, nult, Glt, ρ=0.0, CO2=0.0)
    nutt = nult * (1 - nult * El / Et) / (1 - nult)
    S = [1/El -nult/El -nult/El 0 0 0;
        -nult/El 1/Et -nutt/Et 0 0 0;
        -nult/El -nutt/Et 1/Et 0 0 0;
        0 0 0 2*(1+nutt)/Et 0 0;
        0 0 0 0 1/Glt 0;
        0 0 0 0 0 1/Glt]
    C = fromvoigt(SymmetricTensor{4,3,Float64}, inv(S))
    return Material{3,Float64,typeof(C)}(C, ρ, CO2)
end
