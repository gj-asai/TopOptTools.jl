using Ferrite, FerriteGmsh
using TopOpt

function mbb_simp(volfrac, rρ)
    # mesh
    grid = togrid("test/models/mbb.msh")
    addfacetset!(grid, "symmetry", x -> x[1] ≈ 0.0) # left edge
    addnodeset!(grid, "support", x -> x[1] ≈ 100.0 && x[2] ≈ 0.0) # bottom right corner
    addnodeset!(grid, "force", x -> x[1] ≈ 0.0 && x[2] ≈ 40.0) # top left corner

    # material
    mat = Isotropic2D(E=1, nu=0.3)
    mat_interp = SIMP(mat, 3.0)

    # FE model
    model = TopOpt.FEModel(
        grid=grid,
        ip=Lagrange{RefQuadrilateral,1}(), # linear elements
        qr=QuadratureRule{RefQuadrilateral}(2), # 2 point quadrature
        mat_interp=mat_interp,
        constraints=[
            Dirichlet(:u, getfacetset(grid, "symmetry"), (x, t) -> 0.0, [1]), # block x displacement
            Dirichlet(:u, getnodeset(grid, "support"), (x, t) -> 0.0, [2]), # block y displacement
        ],
        loads=[
            TopOpt.NodalLoad("force", (0.0, -1.0)),
        ],
    )
    density_filter = ConvolutionFilter(rρ, model)

    results = FEResults(model)

    # Defining initial values
    num_elem = getncells(model.grid)
    x0 = fill(volfrac, num_elem)
    xmin = fill(1e-3, num_elem)
    xmax = fill(1, num_elem)

    # Objective: compliance
    function objective(x)
        fea!(results, x, model)
        return compliance(x, results, model)
    end
    function dobjective(x)
        dcdx = dcompliance(x, results, model)
        TopOpt.filter!(dcdx, x, density_filter)
        return dcdx
    end

    # Constraint: max volume fraction
    constraint(x) = TopOpt.volume(x, results, model) / volfrac - 1
    dconstraint(x) = dvolume(x, results, model) / volfrac

    # Post: do nothing
    post(solution; kwargs...) = nothing

    # Run optimization
    opts = OptimOpts(maxiter=500, reltol=1e-5)
    x = topopt(objective, dobjective, constraint, dconstraint, x0, xmin, xmax, model, opts, post)

    nothing
end
