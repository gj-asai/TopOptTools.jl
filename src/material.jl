struct Material{dim}
    C::SymmetricTensor{4,dim}
    ρ::Float64
    CO2::Float64
end

function Isotropic2D(; E, nu, ρ=0.0, CO2=0.0)
    S = [1/E -nu/E 0;
        -nu/E 1/E 0;
        0 0 2(1+nu)/E]
    C = fromvoigt(SymmetricTensor{4,2}, inv(S))
    return Material(C, ρ, CO2)
end

function Orthotropic2D(; El, Et, nult, Glt, ρ=0.0, CO2=0.0)
    S = [1/El -nult/El 0;
        -nult/El 1/Et 0;
        0 0 1/Glt]
    C = fromvoigt(SymmetricTensor{4,2}, inv(S))
    return Material(C, ρ, CO2)
end
