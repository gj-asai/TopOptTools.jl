using Ferrite, FerriteGmsh
using Random

using TopOpt

function mbb_somp(volfrac, rρ, rθ, seed=1234)
    Random.seed!(seed)

    # mesh
    grid = togrid("test/models/mbb.msh")
    addfacetset!(grid, "symmetry", x -> x[1] ≈ 0.0) # left edge
    addnodeset!(grid, "support", x -> x[1] ≈ 100.0 && x[2] ≈ 0.0) # bottom right corner
    addnodeset!(grid, "force", x -> x[1] ≈ 0.0 && x[2] ≈ 40.0) # top left corner

    # material
    carbon = Orthotropic2D(El=122.98e3, Et=2.88e3, nult=0.25, Glt=1.23e3, ρ=1.54e-3, CO2=11.29)
    mat_interp = SOMP(carbon, 3.0)

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
            TopOpt.NodalLoad("force", (0.0, -100.0)),
        ],
    )
    density_filter = ConvolutionFilter(rρ, model)
    orientation_filter = ConvolutionFilter(rθ, model)

    results = FEResults(model)

    # Defining initial values
    num_elem = getncells(model.grid)
    x0 = hcat(fill(volfrac, num_elem), π * rand(num_elem) .- π / 2)'[:]
    xmin = hcat(fill(1e-3, num_elem), fill(-π, num_elem))'[:]
    xmax = hcat(fill(1, num_elem), fill(π, num_elem))'[:]

    # Objective: compliance
    function objective(x)
        x[2:2:end] .= tan.(atan.(x[2:2:end]))
        @views TopOpt.filter!(x[2:2:end], orientation_filter)

        fea!(results, x, model)
        return compliance(x, results, model)
    end
    function dobjective(x)
        dcdx = dcompliance(x, results, model)
        @views TopOpt.filter!(dcdx[1:2:end], x[1:2:end], density_filter)
        return dcdx
    end

    # Constraint: max volume fraction
    constraint(x) = TopOpt.volume(x, results, model) / volfrac - 1
    dconstraint(x) = dvolume(x, results, model) / volfrac

    # After each iteration: filter the orientations
    function post(solution; kwargs...)
        x = solution.x
        x[2:2:end] .= atan.(tan.(x[2:2:end]))
        @views TopOpt.filter!(x[2:2:end], orientation_filter)
    end

    # Run optimization
    opts = OptimOpts(maxiter=500, reltol=1e-6)
    x = topopt(objective, dobjective, constraint, dconstraint, x0, xmin, xmax, model, opts, post)

    nothing
end
