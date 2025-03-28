using Ferrite, FerriteGmsh
using TimerOutputs
using WriteVTK, JLD2
using Random
using Printf

using TopOpt

function mbb_minimpact_mmsomp3d(volfrac, rρ, rθ, wimpact; echo=true, maxiter=500, seed=1234, filename=nothing)
    Random.seed!(seed)
    reset_timer!()
    !isnothing(filename) && (pvd = paraview_collection(filename))

    # Import mesh
    @timeit "read mesh" begin
        grid = togrid("test/models/mbb3d.msh")
        addfacetset!(grid, "symmetry", x -> x[1] ≈ 0.0) # left face
        addnodeset!(grid, "support", x -> x[1] ≈ 100.0 && x[2] ≈ 0.0) # bottom right edge
        addnodeset!(grid, "force", x -> x[1] ≈ 0.0 && x[2] ≈ 40.0) # top left edge
    end

    @timeit "build model" begin
        # Define materials
        carbon = Orthotropic3D(El=122.98e3, Et=2.88e3, nult=0.25, Glt=1.23e3, ρ=1.54e-3, CO2=11.29)
        bamboo = Orthotropic3D(El=10.48e3, Et=2.88e3, nult=0.39, Glt=0.63e3, ρ=0.98e-3, CO2=1.668)
        mat_interp = MMSOMP([carbon, bamboo], 1.0)

        # FE model
        model = TopOpt.FEModel(
            grid=grid,
            ip=Lagrange{RefHexahedron,1}(), # linear elements
            qr=QuadratureRule{RefHexahedron}(2), # 2 point quadrature
            mat_interp=mat_interp,
            constraints=[
                Dirichlet(:u, getfacetset(grid, "symmetry"), (x, t) -> [0.0, 0.0], [1, 3]), # block x displacement
                Dirichlet(:u, getnodeset(grid, "support"), (x, t) -> [0.0, 0.0], [2, 3]), # block y displacement
            ],
            loads=[
                TopOpt.NodalLoad("force", (0.0, -100.0, 0.0)),
            ],
        )

        results = FEResults(model)
        nvar = TopOpt.get_nvar(model)
        nmaterials = nvar - 1

        # Defining initial values
        x0 = zeros(nvar * getncells(model.grid))
        xmin = zeros(nvar * getncells(model.grid))
        xmax = zeros(nvar * getncells(model.grid))
        for i = 1:nmaterials
            x0[i:nvar:end] .= fill(volfrac / nmaterials, getncells(model.grid))
            xmin[i:nvar:end] .= fill(1e-3, getncells(model.grid))
            xmax[i:nvar:end] .= fill(1, getncells(model.grid))
        end
        x0[nmaterials+1:nvar:end] .= π * rand(getncells(model.grid)) .- π / 2
        xmin[nmaterials+1:nvar:end] .= fill(-π, getncells(model.grid))
        xmax[nmaterials+1:nvar:end] .= fill(π, getncells(model.grid))
    end

    @timeit "build filters" begin
        density_filter = ConvolutionFilter(rρ, model)
        orientation_filter = ConvolutionFilter(rθ, model)
    end

    # Obtaining normalization factors for the objective function
    @info "Evaluating initial design"
    fea!(results, x0, model)
    comp_ini = compliance(x0, results, model) * volfrac^model.mat_interp.penal
    CO2_ini = impact(x0, results, model)

    # Objective function: (1-w).compliance/c_0 + w.impact/CO2_0
    function objective(x)
        x[nmaterials+1:nvar:end] .= atan.(tan.(x[nmaterials+1:nvar:end]))
        @views TopOpt.filter!(x[nmaterials+1:nvar:end], orientation_filter)
        fea!(results, x, model)

        return (1 - wimpact) * compliance(x, results, model) / comp_ini + wimpact * impact(x, results, model) / CO2_ini
    end
    function dobjective(x)
        dcdx = dcompliance(x, results, model)
        dCO2dx = dimpact(x, results, model)

        # Filter sensitivities
        for i = 1:nmaterials
            @views TopOpt.filter!(dcdx[i:nvar:end], x[i:nvar:end], density_filter)
            @views TopOpt.filter!(dCO2dx[i:nvar:end], x[i:nvar:end], density_filter)
        end

        return (1 - wimpact) * dcdx / comp_ini + wimpact * dCO2dx / CO2_ini
    end

    # Constraint: max volume fraction
    constraint(x) = TopOpt.volume(x, results, model) / volfrac - 1
    dconstraint(x) = dvolume(x, results, model) / volfrac

    # After each iteration:
    # - Filter orientations
    # - Print current state
    # - Save current state to history dictionary
    # - Apply continuation
    history = Dict(
        :final_x => Float64[],
        :compliance => Float64[],
        :impact => Float64[],
        :objective => Float64[],
        :constraint => Vector{Float64}[],
        :final_compliance => 0,
    )
    function post(solution; kwargs...)
        # Filter orientations
        x = solution.x
        x[nmaterials+1:nvar:end] .= atan.(tan.(x[nmaterials+1:nvar:end]))
        @views TopOpt.filter!(x[nmaterials+1:nvar:end], orientation_filter)

        get(kwargs, :update, false) && return

        # Print current state
        comp = compliance(x, results, model)
        CO2 = impact(x, results, model)
        volfracs = Float64[]
        for i = 1:nmaterials
            push!(volfracs, x[i:nvar:end] ⋅ model.elemvol / sum(model.elemvol))
        end
        @info "c = $(round(comp, sigdigits=4))\tCO2 = $(round(CO2, sigdigits=4))\t volfracs = $(round.(100*volfracs, digits=2))"

        # Push to history
        history[:final_x] = x
        push!(history[:compliance], comp)
        push!(history[:impact], CO2)
        push!(history[:objective], solution.f)
        push!(history[:constraint], solution.g)

        # Save iteration
        !isnothing(filename) && @timeit "export" begin
            i = length(history[:compliance]) - 1
            filename_i = @sprintf "%s.%4.4d.vtu" filename i
            VTKGridFile(filename_i, grid) do vtk
                write_cell_data(vtk, @view(x[1:3:end]), "carbon")
                write_cell_data(vtk, @view(x[2:3:end]), "bamboo")
                write_cell_data(vtk, @view(x[3:3:end]), "theta")
                pvd[i] = vtk
            end
        end

        # Continuation up to p = 5
        Δfrel = abs(solution.f - solution.prevf) / solution.prevf
        if Δfrel < 5e-4 && Δfrel > opts.reltol && model.mat_interp.penal < 5.0
            model.mat_interp.penal += 1.0
            @info "Updated p to $(model.mat_interp.penal)"
        end
    end

    # Run optimization
    opts = OptimOpts(maxiter=maxiter, reltol=1e-5)
    try
        x = topopt(objective, dobjective, constraint, dconstraint, x0, xmin, xmax, model, opts, post)

        # Evaluate final design
        model.mat_interp.penal = 3.0
        fea!(results, x, model)
        history[:final_compliance] = TopOpt.compliance(x, results, model)
        @info "p = 3 equivalent compliance: $(round(history[:final_compliance], sigdigits=4))"
    catch e
        # stop optim and save after receiving a SIGINT
        e isa InterruptException || e isa TaskFailedException || rethrow()
        @warn "Computation interrupted"
        x = history[:final_x]
    end
    merge!(TimerOutputs.get_defaulttimer(), TopOpt.timer)

    # Save
    !isnothing(filename) && @timeit "export" begin
        # history
        save("$(filename).jld2", "history", history)
        @info "Saved file $(filename).jld2"

        # paraview .pvd
        vtk_save(pvd)
        @info "Saved file $(filename).pvd"
    end

    echo && print_timer(linechars=:ascii)

    nothing
end
