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

@testset "Rotate orthotropic 2D" begin
    El = 2.0
    Et = 1.0
    nult = 0.3
    Glt = 1.0

    interp = SOMP(Orthotropic2D(; El, Et, nult, Glt), 3.0)
    rotated_C = TopOpt.interpolate([1.0, deg2rad(30)], interp)

    C = tovoigt(Orthotropic2D(; El, Et, nult, Glt).C)
    c = cosd(30)
    s = sind(30)
    Cnew11 = C[1, 1] * c^4 + 2 * (C[1, 2] + 2 * C[3, 3]) * s^2 * c^2 + C[2, 2] * s^4
    Cnew12 = (C[1, 1] + C[2, 2] - 4 * C[3, 3]) * s^2 * c^2 + C[1, 2] * (s^4 + c^4)
    Cnew22 = C[1, 1] * s^4 + 2 * (C[1, 2] + 2 * C[3, 3]) * s^2 * c^2 + C[2, 2] * c^4
    Cnew16 = (C[1, 1] - C[1, 2] - 2 * C[3, 3]) * s * c^3 + (C[1, 2] - C[2, 2] + 2 * C[3, 3]) * s^3 * c
    Cnew26 = (C[1, 1] - C[1, 2] - 2 * C[3, 3]) * s^3 * c + (C[1, 2] - C[2, 2] + 2 * C[3, 3]) * s * c^3
    Cnew66 = (C[1, 1] + C[2, 2] - 2 * C[1, 2] - 2 * C[3, 3]) * s^2 * c^2 + C[3, 3] * (s^4 + c^4)

    @test tovoigt(rotated_C) ≈ [Cnew11 Cnew12 Cnew16; Cnew12 Cnew22 Cnew26; Cnew16 Cnew26 Cnew66]
end
