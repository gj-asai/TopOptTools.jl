# all expected values from
# https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python

@testset "Filtering a Vector" begin
    # 3x2 mbb
    model = FEModel(
        grid=generate_grid(Quadrilateral, (3, 2), Vec((0.0, 0.0)), Vec((3.0, 2.0))),
        ip=Lagrange{RefQuadrilateral,1}(), # linear elements
        qr=QuadratureRule{RefQuadrilateral}(2), # 2 point quadrature
        mat_interp=SIMP(Isotropic2D(E=1.0, nu=0.3), 3.0),
        constraints=Dirichlet[],
        loads=NodalLoad{2,Float64}[],
    )
    f = ConvolutionFilter(1.5, model)

    cell_order = [2, 4, 6, 1, 3, 5]
    x = [0.05837059, 0.27439468, 0.76653813, 0.29748424, 0.24750964, 0.24884174][cell_order]
    dc = [0.98923353, 0.90642167, 0.238167, 0.09401264, 0.46788691, 0.13252324][cell_order]
    expected_filtered = [2.01845418, 0.60828234, 0.16402637, 0.30627508, 0.44355712, 0.2129366][cell_order]

    TopOpt.filter!(dc, x, f)
    @test dc ≈ expected_filtered atol = 1e-7
end

@testset "Filtering a DesignVariables" begin
    # 3x2 mbb
    model = FEModel(
        grid=generate_grid(Quadrilateral, (3, 2), Vec((0.0, 0.0)), Vec((3.0, 2.0))),
        ip=Lagrange{RefQuadrilateral,1}(), # linear elements
        qr=QuadratureRule{RefQuadrilateral}(2), # 2 point quadrature
        mat_interp=SIMP(Isotropic2D(E=1.0, nu=0.3), 3.0),
        constraints=Dirichlet[],
        loads=NodalLoad{2,Float64}[],
    )
    f = ConvolutionFilter(1.5, model)

    cell_order = [2, 4, 6, 1, 3, 5]
    x = [0.05837059, 0.27439468, 0.76653813, 0.29748424, 0.24750964, 0.24884174][cell_order]
    dc = [0.98923353, 0.90642167, 0.238167, 0.09401264, 0.46788691, 0.13252324][cell_order]
    expected_filtered = [2.01845418, 0.60828234, 0.16402637, 0.30627508, 0.44355712, 0.2129366][cell_order]

    x_dv = DesignVariables(1)
    for xe in x
        push!(x_dv, xe, 1e-3, 1)
    end
    TopOpt.filter!(dc, x, f)
    @test dc ≈ expected_filtered atol = 1e-7
end
