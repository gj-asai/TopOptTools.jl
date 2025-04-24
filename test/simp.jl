# expected value from
# Sigmund, O. A 99 line topology optimization code written in Matlab. Struct Multidisc Optim 21, 120–127 (2001). https://doi.org/10.1007/s001580050176
# top(60, 20, 0.5, 3.0, 1.5)

using Ferrite

@testset "SIMP" begin
    # 60x20 mbb
    grid = generate_grid(Quadrilateral, (60, 20), Vec((0.0, 0.0)), Vec((60.0, 20.0)))
    addfacetset!(grid, "symmetry", x -> x[1] ≈ 0.0) # left edge
    addnodeset!(grid, "support", x -> x[1] ≈ 60.0 && x[2] ≈ 0.0) # bottom right corner
    addnodeset!(grid, "force", x -> x[1] ≈ 0.0 && x[2] ≈ 20.0) # top left corner

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
    results = FEResults(model)

    x0 = DesignVariables(1)
    foreach(1:1200) do _
        push!(x0, 1.0, 1e-3, 1.0)
    end

    density_filter = ConvolutionFilter(1.5, model)

    function objective(x)
        fea!(results, x, model)
        return compliance(x, results, model)
    end
    function dobjective(x)
        dcdx = dcompliance(x, results, model)
        TopOpt.filter!(dcdx, x, density_filter)
        return dcdx
    end
    obj = MMA.Objective(objective, dobjective)

    constraint(x) = TopOpt.volume(x, results, model) / 0.5 - 1
    dconstraint(x) = dvolume(x, results, model) / 0.5
    cons = MMA.Constraints(constraint, dconstraint)

    opts = MMA.OptimOpts(maxiter=100, reltol=1e-4)
    sol = MMA.optimize(x0, obj, cons; opts)

    @test sol.cur_obj ≈ 203.3061 rtol=1e-3
end
