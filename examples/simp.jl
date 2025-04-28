using Ferrite, FerriteGmsh
using TimerOutputs
using WriteVTK, JLD2
using Printf
using UnPack

using TopOpt

struct SIMP{dim,T<:Real,CT} <: MaterialInterpolation{1,T}
    mat::Material{dim,T,CT}
    penal::T
end

function TopOpt.interpolate(xe::AbstractVector, interp::SIMP)
    ρ = xe[1]
    return ρ^interp.penal * interp.mat.C
end

function compliance(_, results::FEResults{T}, ::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SIMP}
    @unpack K, u = results
    return u' * K * u
end

function dcompliance(_, results::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SIMP}
    @unpack u, ∂Ke∂x = results
    dcdx = zeros(length(∂Ke∂x))
    for cell in CellIterator(model.dh)
        ue = u[celldofs(cell)]
        dcdx[cellid(cell)] = -ue' * ∂Ke∂x[cellid(cell)] * ue
    end
    return dcdx
end

function volume(x, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SIMP}
    return x ⋅ model.elemvol / sum(model.elemvol)
end

function dvolume(_, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SIMP}
    return model.elemvol / sum(model.elemvol)
end

function mbb_simp(volfrac, rρ; echo=true, maxiter=500, filename=nothing, save_partial=false)
    @info "SIMP with volfrac=$volfrac, rρ=$rρ"
    reset_timer!()
    !isnothing(filename) && (pvd = paraview_collection(filename))

    @timeit "read mesh" begin
        mesh_file = "examples/models/mbb.msh"
        grid = redirect_stdout(devnull) do
            togrid(mesh_file)
        end
        addfacetset!(grid, "symmetry", x -> x[1] ≈ 0.0) # left edge
        addnodeset!(grid, "support", x -> x[1] ≈ 100.0 && x[2] ≈ 0.0) # bottom right corner
        addnodeset!(grid, "force", x -> x[1] ≈ 0.0 && x[2] ≈ 40.0) # top left corner
        @info "Done reading $(mesh_file): $(getnnodes(grid)) nodes, $(getncells(grid)) elements"
    end

    @timeit "build model" begin
        # Define materials
        mat = Isotropic2D(E=1e3, nu=0.3)
        mat_interp = SIMP(mat, 3.0)

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

        # Defining initial values
        x0 = DesignVariables(1)
        foreach(1:num_elem) do _
            push!(x0, volfrac, 1e-3, 1)
        end
    end

    @timeit "build filters" begin
        @info "Build density filter"
        density_filter = ConvolutionFilter(rρ, model)
    end

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
    obj = MMA.Objective(objective, dobjective)

    # Constraint: max volume fraction
    constraint(x) = volume(x, results, model) / volfrac - 1
    dconstraint(x) = dvolume(x, results, model) / volfrac
    cons = MMA.Constraints(constraint, dconstraint)

    # After each iteration:
    # - Print current state
    # - Save current state to history dictionary
    # - Apply continuation
    history = Dict(
        :final_x => Float64[],
        :final_u => Float64[],
        :compliance => Float64[],
        :penal => Float64[],
        :objective => Float64[],
        :constraint => Vector{Float64}[],
        :final_compliance => 0,
    )
    function post(mma_state)
        @info @sprintf "It = %4d | c = %10.4f" mma_state.it mma_state.cur_obj

        # Push to history
        x = mma_state.x
        comp = compliance(x, results, model)

        history[:final_x] = x
        history[:final_u] = results.u
        push!(history[:compliance], comp)
        push!(history[:penal], mat_interp.penal)
        push!(history[:objective], mma_state.cur_obj)
        push!(history[:constraint], mma_state.cur_cons)

        # Save iteration
        !isnothing(filename) && save_partial && @timeit "export" begin
                i = length(history[:compliance]) - 1
                filename_i = @sprintf "%s.%4.4d.vtu" filename i
                VTKGridFile(filename_i, grid) do vtk
                    write_cell_data(vtk, x, "carbon")
                    pvd[i] = vtk
                end
            end
    end

    # Run optimization
    opts = MMA.OptimOpts(maxiter=maxiter, reltol=1e-5, asydecr=0.3, asyincr=1.1)
    x = try
        @info "Starting optimization with p = $(model.mat_interp.penal)"
        sol = MMA.optimize(x0, obj, cons; post, opts)
        sol.x
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        history[:final_x]
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

        # final result
        VTKGridFile("$(filename).final.vtu", grid) do vtk
            write_cell_data(vtk, x, "carbon")
            write_solution(vtk, model.dh, history[:final_u])
        end
        @info "Saved file $(filename).final.vtu"
    end

    echo && print_timer(linechars=:ascii)

    return history
end
