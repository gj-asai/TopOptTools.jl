using Ferrite, FerriteGmsh
using TimerOutputs
using WriteVTK, JLD2
using Printf

using TopOpt

function mbb_minimpact_somp(volfrac, rρ, rθ, wimpact; echo=true, maxiter=2500, angle=0, filename=nothing)
    reset_timer!()
    !isnothing(filename) && (pvd = paraview_collection(filename))

    # Import mesh
    @timeit "read mesh" begin
        grid = togrid("test/models/mbb.msh")
        addfacetset!(grid, "symmetry", x -> x[1] ≈ 0.0) # left edge
        addnodeset!(grid, "support", x -> x[1] ≈ 100.0 && x[2] ≈ 0.0) # bottom right corner
        addnodeset!(grid, "force", x -> x[1] ≈ 0.0 && x[2] ≈ 40.0) # top left corner
    end

    @timeit "build model" begin
        # Define materials
        mat = Orthotropic2D(El=50e3, Et=5e3, nult=0.30, Glt=1e3, ρ=2e-3, CO2=10.0)
        mat_interp = SOMP(mat, 1.0)

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

        results = FEResults(model)
        num_elem = getncells(model.grid)

        # Defining initial values
        x0 = zeros(2 * num_elem)
        xmin = zeros(2 * num_elem)
        xmax = zeros(2 * num_elem)

        x0[1:2:end] .= fill(volfrac, num_elem)
        xmin[1:2:end] .= fill(1e-3, num_elem)
        xmax[1:2:end] .= fill(1, num_elem)

        x0[2:2:end] .= fill(deg2rad(angle), num_elem)
        xmin[2:2:end] .= fill(-π, num_elem)
        xmax[2:2:end] .= fill(π, num_elem)
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
        x[2:2:end] .= atan.(tan.(x[2:2:end]))
        @views TopOpt.filter!(x[2:2:end], orientation_filter)
        fea!(results, x, model)

        return (1 - wimpact) * compliance(x, results, model) / comp_ini + wimpact * impact(x, results, model) / CO2_ini
    end
    function dobjective(x)
        dcdx = dcompliance(x, results, model)
        dCO2dx = dimpact(x, results, model)

        # Filter sensitivities
        @views TopOpt.filter!(dcdx[1:2:end], x[1:2:end], density_filter)
        @views TopOpt.filter!(dCO2dx[1:2:end], x[1:2:end], density_filter)

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
        :penal => Float64[],
        :objective => Float64[],
        :constraint => Vector{Float64}[],
        :final_compliance => 0,
    )
    function post(solution; kwargs...)
        # Filter orientations
        x = solution.x
        x[2:2:end] .= atan.(tan.(x[2:2:end]))
        @views TopOpt.filter!(x[2:2:end], orientation_filter)

        get(kwargs, :update, false) && return

        # Print current state
        comp = compliance(x, results, model)
        CO2 = impact(x, results, model)
        @info "c = $(round(comp, sigdigits=4))\tCO2 = $(round(CO2, sigdigits=4))"

        # Push to history
        history[:final_x] = x
        push!(history[:compliance], comp)
        push!(history[:impact], CO2)
        push!(history[:penal], mat_interp.penal)
        push!(history[:objective], solution.f)
        push!(history[:constraint], solution.g)

        # Save iteration
        !isnothing(filename) && @timeit "export" begin
            i = length(history[:compliance]) - 1
            filename_i = @sprintf "%s.%4.4d.vtu" filename i
            VTKGridFile(filename_i, grid) do vtk
                write_cell_data(vtk, @view(x[1:2:end]), "carbon")
                write_cell_data(vtk, @view(x[2:2:end]), "theta")
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
    opts = OptimOpts(maxiter=maxiter, reltol=1e-6)
    try
        x = topopt(objective, dobjective, constraint, dconstraint, x0, xmin, xmax, model, opts, post)

        # Evaluate final design
        model.mat_interp.penal = 3.0
        fea!(results, x, model)
        history[:final_compliance] = TopOpt.compliance(x, results, model)
        @info "p = 3 equivalent compliance: $(round(history[:final_compliance], sigdigits=4))"
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        x = history[:final_x]
        history[:final_compliance] = NaN
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

    echo && print_timer(linechars=:ascii, title="w=$(wimpact), angle=$(angle)")

    return history
end
