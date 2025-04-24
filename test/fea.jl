# all expected values from
# https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python

using Ferrite

@testset "Element stiffness" begin
    # generate a mesh with a single element
    model = FEModel(
        grid=generate_grid(Quadrilateral, (1, 1), Vec((0.0, 0.0)), Vec((1.0, 1.0))),
        ip=Lagrange{RefQuadrilateral,1}(), # linear elements
        qr=QuadratureRule{RefQuadrilateral}(2), # 2 point quadrature
        mat_interp=SIMP(Isotropic2D(E=1.0, nu=0.3), 3.0),
        constraints=Dirichlet[],
        loads=NodalLoad{2,Float64}[],
    )
    Ke = zeros(8, 8)

    # density = 1.0
    expected_Ke = [
        0.49450549 0.17857143 -0.3021978 -0.01373626 -0.24725275 -0.17857143 0.05494505 0.01373626;
        0.17857143 0.49450549 0.01373626 0.05494505 -0.17857143 -0.24725275 -0.01373626 -0.3021978;
        -0.3021978 0.01373626 0.49450549 -0.17857143 0.05494505 -0.01373626 -0.24725275 0.17857143;
        -0.01373626 0.05494505 -0.17857143 0.49450549 0.01373626 -0.3021978 0.17857143 -0.24725275;
        -0.24725275 -0.17857143 0.05494505 0.01373626 0.49450549 0.17857143 -0.3021978 -0.01373626;
        -0.17857143 -0.24725275 -0.01373626 -0.3021978 0.17857143 0.49450549 0.01373626 0.05494505;
        0.05494505 -0.01373626 -0.24725275 0.17857143 -0.3021978 0.01373626 0.49450549 -0.17857143;
        0.01373626 -0.3021978 0.17857143 -0.24725275 -0.01373626 0.05494505 -0.17857143 0.49450549
    ]
    TopOpt.element_stiffness!(Ke, [1.0], model.cellvalues, model)
    @test Ke ≈ expected_Ke atol = 1e-7

    # density = 0.5
    expected_Ke = [
        0.06181319 0.02232143 -0.03777473 -0.00171703 -0.03090659 -0.02232143 0.00686813 0.00171703;
        0.02232143 0.06181319 0.00171703 0.00686813 -0.02232143 -0.03090659 -0.00171703 -0.03777473;
        -0.03777473 0.00171703 0.06181319 -0.02232143 0.00686813 -0.00171703 -0.03090659 0.02232143;
        -0.00171703 0.00686813 -0.02232143 0.06181319 0.00171703 -0.03777473 0.02232143 -0.03090659;
        -0.03090659 -0.02232143 0.00686813 0.00171703 0.06181319 0.02232143 -0.03777473 -0.00171703;
        -0.02232143 -0.03090659 -0.00171703 -0.03777473 0.02232143 0.06181319 0.00171703 0.00686813;
        0.00686813 -0.00171703 -0.03090659 0.02232143 -0.03777473 0.00171703 0.06181319 -0.02232143;
        0.00171703 -0.03777473 0.02232143 -0.03090659 -0.00171703 0.00686813 -0.02232143 0.06181319
    ]
    TopOpt.element_stiffness!(Ke, [0.5], model.cellvalues, model)
    @test Ke ≈ expected_Ke atol = 1e-7
end

@testset "FEA" begin
    # 2x2 mbb
    grid = generate_grid(Quadrilateral, (2, 2), Vec((0.0, 0.0)), Vec((2.0, 2.0)))
    addfacetset!(grid, "symmetry", x -> x[1] ≈ 0.0) # left edge
    addnodeset!(grid, "support", x -> x[1] ≈ 2.0 && x[2] ≈ 0.0) # bottom right corner
    addnodeset!(grid, "force", x -> x[1] ≈ 0.0 && x[2] ≈ 2.0) # top left corner

    # FE model
    model = FEModel(
        grid=grid,
        ip=Lagrange{RefQuadrilateral,1}(), # linear elements
        qr=QuadratureRule{RefQuadrilateral}(2), # 2 point quadrature
        mat_interp=SIMP(Isotropic2D(E=1.0, nu=0.3), 3.0),
        constraints=[
            Dirichlet(:u, getfacetset(grid, "symmetry"), (x, t) -> 0.0, [1]), # block x displacement
            Dirichlet(:u, getnodeset(grid, "support"), (x, t) -> 0.0, [2]), # block y displacement
        ],
        loads=[
            TopOpt.NodalLoad("force", (0.0, -1.0)),
        ],
    )
    x = DesignVariables(1)
    foreach(1:4) do _
        push!(x, 1.0, 1e-3, 1.0)
    end

    results = FEResults(model)
    fea!(results, x, model)
    comp = compliance(x, results, model)
    dcdx = dcompliance(x, results, model)

    expected_comp = 8.584462644544166
    expected_u = [0.00000000e+00, -8.58446264e+00, 0.00000000e+00, -7.20663690e+00, 0.00000000e+00, -6.46734057e+00, -1.68440742e+00, -4.90278793e+00, 1.03541625e-01, -5.23751925e+00, 2.03206317e+00, -5.17077999e+00, -2.07820303e+00, -2.41886204e+00, 1.73244272e-03, -1.93016631e+00, 3.27473815e+00, 0.00000000e+00]
    dof_order = [5, 6, 11, 12, 9, 10, 3, 4, 17, 18, 15, 16, 7, 8, 1, 2, 13, 14] # translates dofs in expected to dofs in model

    # derivative of the compliance via finite differences
    delta = 1e-8
    dcdx_fd = zeros(4)
    for i in eachindex(x)
        x_left = copy(x)
        x_left[i] -= delta
        fea!(results, x_left, model)
        comp_left = compliance(x_left, results, model)

        x_right = copy(x)
        x_right[i] += delta
        fea!(results, x_right, model)
        comp_right = compliance(x_right, results, model)

        dcdx_fd[i] = (comp_right - comp_left) / 2delta
    end

    @test results.u ≈ expected_u[dof_order]
    @test comp ≈ expected_comp
    @test dcompliance(x, results, model) ≈ dcdx_fd atol = 1e-4
end

@testset "SOMP AD" begin
    # 5x3 mbb
    grid = generate_grid(Quadrilateral, (5, 3), Vec((0.0, 0.0)), Vec((5.0, 3.0)))
    addfacetset!(grid, "symmetry", x -> x[1] ≈ 0.0) # left edge
    addnodeset!(grid, "support", x -> x[1] ≈ 5.0 && x[2] ≈ 0.0) # bottom right corner
    addnodeset!(grid, "force", x -> x[1] ≈ 0.0 && x[2] ≈ 3.0) # top left corner

    # FE model
    model = FEModel(
        grid=grid,
        ip=Lagrange{RefQuadrilateral,1}(), # linear elements
        qr=QuadratureRule{RefQuadrilateral}(2), # 2 point quadrature
        mat_interp=SOMP(Orthotropic2D(El=2.0, Et=1.0, nult=0.3, Glt=1.0), 3.0),
        constraints=[
            Dirichlet(:u, getfacetset(grid, "symmetry"), (x, t) -> 0.0, [1]), # block x displacement
            Dirichlet(:u, getnodeset(grid, "support"), (x, t) -> 0.0, [2]), # block y displacement
        ],
        loads=[
            TopOpt.NodalLoad("force", (0.0, -1.0)),
        ],
    )
    x = DesignVariables(2)
    foreach(1:15) do _
        push!(x, 0.5, 1e-3, 1.0)
        push!(x, deg2rad(45), -π, π)
    end

    results = FEResults(model)
    fea!(results, x, model)
    dcdx = dcompliance(x, results, model)

    # derivative of the compliance via finite differences
    delta = 1e-6
    dcdx_fd = zeros(30)
    for i in eachindex(x)
        x_left = copy(x)
        x_left[i] -= delta
        fea!(results, x_left, model)
        comp_left = compliance(x_left, results, model)

        x_right = copy(x)
        x_right[i] += delta
        fea!(results, x_right, model)
        comp_right = compliance(x_right, results, model)

        dcdx_fd[i] = (comp_right - comp_left) / 2delta
    end

    @test dcompliance(x, results, model) ≈ dcdx_fd atol = 1e-3
end

@testset "MMSOMP AD" begin
    # 5x3 mbb
    grid = generate_grid(Quadrilateral, (5, 3), Vec((0.0, 0.0)), Vec((5.0, 3.0)))
    addfacetset!(grid, "symmetry", x -> x[1] ≈ 0.0) # left edge
    addnodeset!(grid, "support", x -> x[1] ≈ 5.0 && x[2] ≈ 0.0) # bottom right corner
    addnodeset!(grid, "force", x -> x[1] ≈ 0.0 && x[2] ≈ 3.0) # top left corner

    # FE model
    model = FEModel(
        grid=grid,
        ip=Lagrange{RefQuadrilateral,1}(), # linear elements
        qr=QuadratureRule{RefQuadrilateral}(2), # 2 point quadrature
        mat_interp=MMSOMP(
            [
                Orthotropic2D(El=2.0, Et=1.0, nult=0.3, Glt=1.0),
                Orthotropic2D(El=20.0, Et=1.0, nult=0.3, Glt=1.0),
            ], 3.0),
        constraints=[
            Dirichlet(:u, getfacetset(grid, "symmetry"), (x, t) -> 0.0, [1]), # block x displacement
            Dirichlet(:u, getnodeset(grid, "support"), (x, t) -> 0.0, [2]), # block y displacement
        ],
        loads=[
            TopOpt.NodalLoad("force", (0.0, -1.0)),
        ],
    )
    x = DesignVariables(3)
    foreach(1:15) do _
        push!(x, 0.2, 1e-3, 1.0)
        push!(x, 0.3, 1e-3, 1.0)
        push!(x, deg2rad(45), -π, π)
    end

    results = FEResults(model)
    fea!(results, x, model)
    dcdx = dcompliance(x, results, model)

    # derivative of the compliance via finite differences
    delta = 1e-6
    dcdx_fd = zeros(45)
    for i in eachindex(x)
        x_left = copy(x)
        x_left[i] -= delta
        fea!(results, x_left, model)
        comp_left = compliance(x_left, results, model)

        x_right = copy(x)
        x_right[i] += delta
        fea!(results, x_right, model)
        comp_right = compliance(x_right, results, model)

        dcdx_fd[i] = (comp_right - comp_left) / 2delta
    end

    @test dcompliance(x, results, model) ≈ dcdx_fd atol = 1e-4
end
