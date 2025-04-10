using Ferrite, FerriteGmsh
using TimerOutputs
using WriteVTK, JLD2
using Printf

using TopOpt

function mbb_minimpact_mmsomp(volfrac, rρ, rθ, wimpact; echo=true, maxiter=2500, angle=0, filename=nothing, save_partial=false)
    reset_timer!()
    !isnothing(filename) && save_partial && (pvd = paraview_collection(filename))

    # Import mesh
    @timeit "read mesh" begin
        grid = togrid("test/models/mbb.msh")
        addfacetset!(grid, "symmetry", x -> x[1] ≈ 0.0) # left edge
        addnodeset!(grid, "support", x -> x[1] ≈ 100.0 && x[2] ≈ 0.0) # bottom right corner
        addnodeset!(grid, "force", x -> x[1] ≈ 0.0 && x[2] ≈ 40.0) # top left corner
    end

    @timeit "build model" begin
        # Define materials
        carbon = Orthotropic2D(El=122.98e3, Et=2.88e3, nult=0.25, Glt=1.23e3, ρ=1.54e-3, CO2=11.29)
        bamboo = Orthotropic2D(El=10.48e3, Et=2.88e3, nult=0.39, Glt=0.63e3, ρ=0.98e-3, CO2=1.668)
        # bamboo = Orthotropic2D(El=40e3, Et=1e3, nult=0.39, Glt=0.63e3, ρ=0.98e-3, CO2=1.668)
        mat_interp = MMSOMP([carbon, bamboo], 1.0)

        # FE model
        ip = Lagrange{RefQuadrilateral,1}() # linear elements
        qr = QuadratureRule{RefQuadrilateral}(2) # 2 point quadrature
        model = TopOpt.FEModel(
            grid=grid,
            ip=ip,
            qr=qr,
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
        nvar = TopOpt.get_nvar(model)
        nmaterials = nvar - 1

        # Defining initial values
        x0 = zeros(nvar * num_elem)
        xmin = zeros(nvar * num_elem)
        xmax = zeros(nvar * num_elem)
        for i = 1:nmaterials
            x0[i:nvar:end] .= fill(volfrac / nmaterials, num_elem)
            xmin[i:nvar:end] .= fill(1e-3, num_elem)
            xmax[i:nvar:end] .= fill(1.0, num_elem)
        end
        x0[nmaterials+1:nvar:end] .= fill(deg2rad(angle), num_elem)
        xmin[nmaterials+1:nvar:end] .= fill(-π, num_elem)
        xmax[nmaterials+1:nvar:end] .= fill(π, num_elem)
    end

    @timeit "build filters" begin
        density_filter = ConvolutionFilter(rρ, model)
        orientation_filter = ConvolutionFilter(rθ, model)
    end

    # Obtaining normalization factors for the objective function
    @info "Evaluating initial design"
    xnorm = copy(x0)
    xnorm[nmaterials+1:nvar:end] .= 0
    fea!(results, xnorm, model)
    comp_ini = compliance(xnorm, results, model) * volfrac^model.mat_interp.penal
    CO2_ini = impact(xnorm, results, model)

    # Objective function: (1-w).compliance/c_0 + w.impact/CO2_0
    function objective(x)
        # if length(history[:penal]) > 0 && model.mat_interp.penal != history[:penal][end]
        #     qp_principaldir = stress(results, x, model)[5]
        #     cell_principaldir = [mean([atan(dir[2,1], dir[1,1]) for dir in el]) for el in qp_principaldir]
        #     x[nmaterials+1:nvar:end] = cell_principaldir
        # end
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
    # - Print current state
    # - Save current state to history dictionary
    # - Apply continuation
    history = Dict(
        :final_x => Float64[],
        :final_u => Float64[],
        :compliance => Float64[],
        :impact => Float64[],
        :penal => Float64[],
        :objective => Float64[],
        :constraint => Vector{Float64}[],
        :final_compliance => 0,
    )
    function post(solution; kwargs...)
        x = solution.prevx

        comp = compliance(x, results, model)
        CO2 = impact(x, results, model)
        volfracs = Float64[]
        for i = 1:nmaterials
            push!(volfracs, x[i:nvar:end] ⋅ model.elemvol / sum(model.elemvol))
        end

        # Print current state
        get(kwargs, :update, false) && return
        @info "c = $(round(comp, sigdigits=4))\tCO2 = $(round(CO2, sigdigits=4))\t volfracs = $(round.(100*volfracs, digits=2))"

        # Push to history
        history[:final_x] = x
        history[:final_u] = results.u
        push!(history[:compliance], comp)
        push!(history[:impact], CO2)
        push!(history[:penal], mat_interp.penal)
        push!(history[:objective], solution.f)
        push!(history[:constraint], solution.g)

        # Save iteration
        !isnothing(filename) && save_partial && @timeit "export" begin
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
    x = try
        x = topopt(objective, dobjective, constraint, dconstraint, x0, xmin, xmax, model, opts, post)

        # Evaluate final design
        model.mat_interp.penal = 3.0
        fea!(results, x, model)
        history[:final_compliance] = TopOpt.compliance(x, results, model)
        @info "p = 3 equivalent compliance: $(round(history[:final_compliance], sigdigits=4))"
        x
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        history[:final_compliance] = NaN
        x = history[:final_x]
    end
    merge!(TimerOutputs.get_defaulttimer(), TopOpt.timer)

    # Save
    !isnothing(filename) && @timeit "export" begin
        # history
        save("$(filename).jld2", "history", history)
        @info "Saved file $(filename).jld2"

        # paraview .pvd
        if save_partial
            vtk_save(pvd)
            @info "Saved file $(filename).pvd"
        end

        qp_global, qp_material, qp_vonmises, qp_principalstress, qp_principaldir = stress(results, x, model)
        projector = L2Projector(ip^2, model.grid)

        # final result
        VTKGridFile("$(filename).final.vtu", grid) do vtk
            write_cell_data(vtk, @view(x[1:3:end]), "carbon")
            write_cell_data(vtk, @view(x[2:3:end]), "bamboo")
            write_cell_data(vtk, @view(x[3:3:end]), "theta")
            write_solution(vtk, model.dh, history[:final_u])
            write_projection(vtk, projector, project(projector, qp_global, qr), "stress - global")
            write_projection(vtk, projector, project(projector, qp_material, qr), "stress - material")
            write_projection(vtk, projector, project(projector, qp_vonmises, qr), "stress - von mises")
            write_projection(vtk, projector, project(projector, qp_principalstress, qr), "stress - principal")

            for i = 1:2
                principaldir_i = [[Tensor{1,2}(vectors[:, i]) for vectors in el] for el in qp_principaldir]
                write_projection(vtk, projector, project(projector, principaldir_i, qr), "principal direction $i")
            end
        end
        @info "Saved file $(filename).final.vtu"
    end

    echo && print_timer(linechars=:ascii, title="w=$(wimpact), angle=$(angle)")

    return history
end
