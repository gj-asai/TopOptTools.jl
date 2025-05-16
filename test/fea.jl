# all expected values from
# https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python

using Ferrite, UnPack

struct SIMP{dim,T<:Real,CT} <: MaterialInterpolation{1,T}
    mat::Material{dim,T,CT}
    penal::T
end

function TopOpt.interpolate(xe::AbstractVector, interp::SIMP)
    ρ = xe[1]
    return ρ^interp.penal * interp.mat.C
end

function compliance(results::FEResults)
    @unpack K, u = results
    return u' * K * u
end

function dcompliance(results::FEResults)
    @unpack u, ∂Ke∂x = results
    model = results.model

    dcdx = zeros(length(∂Ke∂x))
    for cell in CellIterator(model.dh)
        ue = u[celldofs(cell)]
        dcdx[cellid(cell)] = -ue' * ∂Ke∂x[cellid(cell)] * ue
    end
    return dcdx
end

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
    x = DesignVector(1)
    foreach(1:4) do _
        push!(x, 1.0, 1e-3, 1.0)
    end

    results = FEResults(x, model)
    fea!(results, x)

    comp = compliance(results)
    dcdx = dcompliance(results)
    u = copy(results.u)

    expected_u = [0.00000000e+00, -8.58446264e+00, 0.00000000e+00, -7.20663690e+00, 0.00000000e+00, -6.46734057e+00, -1.68440742e+00, -4.90278793e+00, 1.03541625e-01, -5.23751925e+00, 2.03206317e+00, -5.17077999e+00, -2.07820303e+00, -2.41886204e+00, 1.73244272e-03, -1.93016631e+00, 3.27473815e+00, 0.00000000e+00]
    expected_comp = 8.584462644544166
    dof_order = [5, 6, 11, 12, 9, 10, 3, 4, 17, 18, 15, 16, 7, 8, 1, 2, 13, 14] # translates dofs in expected to dofs in model

    # derivative of the compliance via finite differences
    delta = 1e-8
    dcdx_fd = zeros(4)
    for i in eachindex(x)
        x_left = copy(x)
        x_left[i] -= delta
        fea!(results, x_left)
        comp_left = compliance(results)

        x_right = copy(x)
        x_right[i] += delta
        fea!(results, x_right)
        comp_right = compliance(results)
        dcdx_fd[i] = (comp_right - comp_left) / 2delta
    end

    @test u ≈ expected_u[dof_order]
    @test comp ≈ expected_comp
    @test dcdx ≈ dcdx_fd atol = 1e-4
end
