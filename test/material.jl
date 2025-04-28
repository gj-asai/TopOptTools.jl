using Tensors

@testset "Isotropic 2D" begin
    E = 1.0
    nu = 0.3

    expected_C = E / (1 - nu^2) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2]
    @test tovoigt(Isotropic2D(; E, nu).C) ≈ expected_C
end

@testset "Orthotropic 2D" begin
    El = 2.0
    Et = 1.0
    nult = 0.3
    Glt = 1.0

    nutl = Et / El * nult
    expected_C = [
        El/(1-nult*nutl) nult*Et/(1-nult*nutl) 0;
        nutl*El/(1-nult*nutl) Et/(1-nult*nutl) 0;
        0 0 Glt
    ]
    @test tovoigt(Orthotropic2D(; El, Et, nult, Glt).C) ≈ expected_C
end
