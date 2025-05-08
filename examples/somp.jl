using Ferrite, FerriteGmsh
using TimerOutputs
using WriteVTK, JLD2
using Printf
using UnPack

using TopOpt

mutable struct SOMP{dim,T<:Real,CT} <: MaterialInterpolation{2,T}
    mat::Material{dim,T,CT}
    penal::T
end

function TopOpt.interpolate(xe::AbstractVector, interp::SOMP)
    ρ, θ = xe
    return ρ^interp.penal * rotate(interp.mat.C, θ)
end

function TopOpt.rotate_stress(global_stress::SymmetricTensor{2}, xe::AbstractVector, ::SOMP)
    θ = xe[2]
    return rotate(global_stress, -θ)
end

function compliance(_, results::FEResults{T}, ::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SOMP}
    @unpack K, u = results
    return u' * K * u
end

function dcompliance(_, results::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SOMP}
    @unpack u, ∂Ke∂x = results
    dcdx = zeros(T, length(∂Ke∂x))
    for cell in CellIterator(model.dh)
        ue = u[celldofs(cell)]
        e = cellid(cell)
        for i in 1:nvar
            dcdx[nvar*(e-1)+i] = -ue' * ∂Ke∂x[nvar*(e-1)+i] * ue
        end
    end
    return dcdx
end

function volume(x, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SOMP}
    ρ = @view x[1:2:end]
    return ρ ⋅ model.elemvol / sum(model.elemvol)
end

function dvolume(x, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SOMP}
    ∂g∂x = zeros(T, length(x))
    ∂g∂x[1:2:end] .= model.elemvol / sum(model.elemvol)
    return ∂g∂x
end

function mbb_somp(volfrac, rρ, rθ; echo=true, maxiter=2500, angle=0, filename=nothing, save_partial=false)
    @info "SOMP with volfrac=$volfrac, rρ=$rρ, rθ=$rθ, angle=$angle"
    reset_timer!()
    !isnothing(filename) && save_partial && (pvd = paraview_collection(filename))

    # Import mesh
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
        mat = Orthotropic2D(El=10.48e3, Et=5.26e3, nult=0.39, Glt=1.89e3, ρ=0.98e-3, CO2=1.668)
        mat_interp = SOMP(mat, 3.0)

        # FE model
        ip = Lagrange{RefQuadrilateral,1}() # linear elements
        qr = QuadratureRule{RefQuadrilateral}(2) # 2 point quadrature
        model = FEModel(
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
        x0 = DesignVariables(2)
        foreach(1:num_elem) do _
            push!(x0, volfrac, 1e-3, 1)
            push!(x0, deg2rad(angle), -π, π)
        end
    end

    @timeit "build filters" begin
        @info "Build density filter"
        density_filter = ConvolutionFilter(rρ, model)
        @info "Build orientation filter"
        orientation_filter = ConvolutionFilter(rθ, model)
    end

    function objective(x)
        @views TopOpt.filter!(x[2:2:end], orientation_filter)
        fea!(results, x, model)

        return compliance(x, results, model)
    end
    function dobjective(x)
        dcdx = dcompliance(x, results, model)
        @views TopOpt.filter!(dcdx[1:2:end], x[1:2:end], density_filter)

        return dcdx
    end
    obj = MMA.Objective(objective, dobjective)

    # Constraint: max volume fraction
    constraint(x) = volume(x, results, model) / volfrac - 1
    dconstraint(x) = dvolume(x, results, model) / volfrac
    cons = MMA.Constraints(constraint, dconstraint)

    history = Dict(
        :final_x => Float64[],
        :final_u => Float64[],
        :compliance => Float64[],
        :penal => Float64[],
        :objective => Float64[],
        :constraint => Vector{Float64}[],
    )

    x = try
        # initialize optimization
        @info "Starting optimization with p = $(model.mat_interp.penal)"
        mma = MMA.MMAProblem(x0, obj, cons)
        fea!(results, mma.state.x, model)

        for _ in 1:maxiter
            # iterate and smooth result before saving
            MMA.iterate(mma)
            @views TopOpt.filter!(mma.state.x[2:2:end], orientation_filter)
            fea!(results, mma.state.x, model)

            comp = compliance(mma.state.x, results, model)
            @info @sprintf "It = %4d | c = %10.4f" mma.state.it mma.state.cur_obj

            # Push to history
            history[:final_x] = mma.state.x
            history[:final_u] = results.u
            push!(history[:compliance], comp)
            push!(history[:penal], mat_interp.penal)
            push!(history[:objective], mma.state.cur_obj)
            push!(history[:constraint], mma.state.cur_cons)

            # Save iteration
            !isnothing(filename) && save_partial && @timeit "export" begin
                    filename_i = @sprintf "%s.%4.4d.vtu" filename mma.state.it
                    VTKGridFile(filename_i, grid) do vtk
                        write_cell_data(vtk, @view(mma.state.x[1:2:end]), "carbon")
                        write_cell_data(vtk, @view(mma.state.x[2:2:end]), "theta")
                        pvd[mma.state.it] = vtk
                    end
                end

            MMA.relative_change(mma.state) < 1e-4 && break
        end

        mma.state.x
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        history[:final_compliance] = NaN
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

        qp_global, qp_material, qp_vonmises, qp_principalstress, qp_principaldir = stress(results, x, model)
        projector = L2Projector(ip^2, model.grid)

        # final result
        VTKGridFile("$(filename).final.vtu", grid) do vtk
            write_cell_data(vtk, @view(x[1:2:end]), "carbon")
            write_cell_data(vtk, @view(x[2:2:end]), "theta")
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

    echo && print_timer(linechars=:ascii, title="angle=$(angle)")

    return history
end
