using Ferrite, FerriteGmsh
using TimerOutputs
using WriteVTK, JLD2
using Printf
using UnPack

using TopOpt

mutable struct MMSOMP{dim,T<:Real,M,N,CT} <: MaterialInterpolation{N,T}
    mat::Vector{Material{dim,T,CT}}
    penal::T
end
MMSOMP(mat::Vector{Material{dim,T,CT}}, penal::T) where {dim,T,CT} = MMSOMP{dim,T,length(mat),length(mat) + 1,CT}(mat, penal)

function TopOpt.interpolate(xe::AbstractVector{T}, interp::MMSOMP{dim}) where {T<:Real,dim}
    ρ, θ = xe[1:end-1], xe[end]
    result = zero(SymmetricTensor{4,dim,T})
    for i in eachindex(interp.mat)
        weight = one(T)
        for j in eachindex(interp.mat)
            if i == j
                weight *= ρ[j]^interp.penal
            else
                weight *= one(T) - ρ[j]^interp.penal
            end
        end
        result += weight * rotate_mmsomp(interp.mat[i], θ)
    end
    result += TopOpt.void(Val(dim)).C
    return result
end
rotate_mmsomp(mat::Material{2}, θ) = rotate(mat.C, θ)
rotate_mmsomp(mat::Material{3}, θ) = rotate(mat.C, Vec{3}((0.0, 0.0, 1.0)), θ)

function TopOpt.rotate_stress(global_stress::SymmetricTensor{2}, xe::AbstractVector, ::MMSOMP)
    θ = xe[end]
    return rotate(global_stress, -θ)
end

function compliance(results::FEResults, ::Type{T} where {T<:MMSOMP})
    @unpack K, u = results
    return u' * K * u
end

function dcompliance(results::FEResults{T}, ::Type{Interp} where {Interp<:MMSOMP}) where {T}
    @unpack u, ∂Ke∂x = results
    model = results.model
    nvar = get_nvar(model)

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

function impact(results::FEResults, ::Type{T} where {T<:MMSOMP})
    x = results.x
    model = results.model
    nvar = get_nvar(model)

    CO2 = 0
    for (i, mati) in enumerate(model.mat_interp.mat)
        ρi = x[i:nvar:end]
        CO2 += mati.CO2 * mati.ρ * ρi ⋅ model.elemvol
    end
    return CO2
end

function dimpact(results::FEResults{T}, ::Type{Interp} where {Interp<:MMSOMP}) where {T}
    x = results.x
    model = results.model
    nvar = get_nvar(model)

    dCO2dx = zeros(T, length(x))
    for (i, mati) in enumerate(model.mat_interp.mat)
        dCO2dx[i:nvar:end] .= mati.CO2 * mati.ρ * model.elemvol
    end
    return dCO2dx
end

function volume(results::FEResults, ::Type{T} where {T<:MMSOMP})
    x = results.x
    model = results.model
    nvar = get_nvar(model)

    ρsum = [sum(x[nvar*(i-1)+1:nvar*i-1]) for i in 1:getncells(model.grid)]
    return ρsum ⋅ model.elemvol / sum(model.elemvol)
end

function dvolume(results::FEResults{T}, ::Type{Interp} where {Interp<:MMSOMP}) where {T}
    x = results.x
    model = results.model
    nvar = get_nvar(model)

    ∂g∂x = zeros(T, length(x))
    for i = 1:length(model.mat_interp.mat)
        ∂g∂x[i:nvar:end] .= model.elemvol / sum(model.elemvol)
    end
    return ∂g∂x
end

function mbb_minimpact_mmsomp(volfrac, rρ, rθ, wimpact; echo=true, maxiter=2500, angle=0, filename=nothing, save_partial=false)
    @info "MMSOMP with volfrac=$volfrac, rρ=$rρ, rθ=$rθ, w=$wimpact, angle=$angle"
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
        carbon = Orthotropic2D(El=122.98e3, Et=10.2e3, nult=0.25, Glt=3.67e3, ρ=1.54e-3, CO2=11.29)
        bamboo = Orthotropic2D(El=10.48e3, Et=5.26e3, nult=0.39, Glt=1.89e3, ρ=0.98e-3, CO2=1.668)
        mat_interp = MMSOMP([carbon, bamboo], 1.0)

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
                NodalLoad("force", (0.0, -100.0)),
            ],
        )

        num_elem = getncells(model.grid)
        nvar = get_nvar(model)
        nmaterials = nvar - 1

        # Defining initial values
        x0 = DesignVector(nvar)
        foreach(1:num_elem) do _
            foreach(1:nmaterials) do _
                push!(x0, volfrac / nmaterials, 1e-3, 1)
            end
            push!(x0, deg2rad(angle), -π, π)
        end

        results = FEResults(x0, model)
    end

    @timeit "build filters" begin
        @info "Build density filter"
        density_filter = ConvolutionFilter(rρ, model)
        @info "Build orientation filter"
        orientation_filter = ConvolutionFilter(rθ, model)
    end

    # Obtaining normalization factors for the objective function
    @info "Evaluating initial design"
    xnorm = copy(x0)
    xnorm[nmaterials+1:nvar:end] .= 0
    fea!(results, xnorm)
    comp_ini = compliance(results, MMSOMP) * volfrac^model.mat_interp.penal
    CO2_ini = impact(results, MMSOMP)
    @info @sprintf "Normalization factors: compliance => %.4f, CO2 => %.4f" comp_ini CO2_ini

    # Objective function: (1-w).compliance/c_0 + w.impact/CO2_0
    function objective(x)
        fea!(results, x)
        return (1 - wimpact) * compliance(results, MMSOMP) / comp_ini + wimpact * impact(results, MMSOMP) / CO2_ini
    end
    function dobjective(x)
        dcdx = dcompliance(results, MMSOMP)
        dCO2dx = dimpact(results, MMSOMP)

        # Filter sensitivities
        for i = 1:nmaterials
            @views TopOpt.filter!(dcdx[i:nvar:end], x[i:nvar:end], density_filter)
            @views TopOpt.filter!(dCO2dx[i:nvar:end], x[i:nvar:end], density_filter)
        end

        return (1 - wimpact) * dcdx / comp_ini + wimpact * dCO2dx / CO2_ini
    end
    obj = MMA.Objective(objective, dobjective)

    # Constraint: max volume fraction
    constraint(_) = volume(results, MMSOMP) / volfrac - 1
    dconstraint(_) = dvolume(results, MMSOMP) / volfrac
    cons = MMA.Constraints(constraint, dconstraint)

    history = Dict(
        :final_x => Float64[],
        :final_u => Float64[],
        :compliance => Float64[],
        :impact => Float64[],
        :penal => Float64[],
        :objective => Float64[],
        :constraint => Vector{Float64}[],
        :volfracs => Vector{Float64}[],
        :final_compliance => 0,
    )

    # initialize optimization
    @info "Starting optimization with p = $(model.mat_interp.penal)"
    mma = MMA.MMAProblem(x0, obj, cons, move=0.5, asyinit=0.1, asydecr=0.5, asyincr=1.1)

    try
        comp = compliance(results, MMSOMP)
        CO2 = impact(results, MMSOMP)
        volfracs = Float64[]
        for i = 1:nmaterials
            push!(volfracs, mma.state.x[i:nvar:end] ⋅ model.elemvol / sum(model.elemvol))
        end

        history[:final_x] = mma.state.x
        history[:final_u] = results.u
        push!(history[:compliance], comp)
        push!(history[:impact], CO2)
        push!(history[:penal], mat_interp.penal)
        push!(history[:objective], mma.state.cur_obj)
        push!(history[:constraint], mma.state.cur_cons)
        push!(history[:volfracs], volfracs)

        inner_it = 0
        for _ in 1:maxiter
            # iterate and smooth result before saving
            MMA.iterate(mma)
            @views TopOpt.filter!(mma.state.x[nmaterials+1:nvar:end], orientation_filter)
            inner_it += 1

            # individual objectives, not expensive because uses the FEA results already calculated
            comp = compliance(results, MMSOMP)
            CO2 = impact(results, MMSOMP)
            volfracs = Float64[]
            for i = 1:nmaterials
                push!(volfracs, mma.state.x[i:nvar:end] ⋅ model.elemvol / sum(model.elemvol))
            end

            # log iteration
            Δf = MMA.relative_change(mma.state, window=10)
            formatted_volfracs = join([@sprintf("%5.2f", 100 * f) for f in volfracs], ", ")
            @info @sprintf "It = %4d | obj = %8.4f | Δobj = %8.2e | c = %10.4f | CO2 = %8.4f | volfracs = [%s] %%" mma.state.it mma.state.cur_obj Δf comp CO2 formatted_volfracs

            # Push to history
            history[:final_x] = mma.state.x
            history[:final_u] = results.u
            push!(history[:compliance], comp)
            push!(history[:impact], CO2)
            push!(history[:penal], mat_interp.penal)
            push!(history[:objective], mma.state.cur_obj)
            push!(history[:constraint], mma.state.cur_cons)
            push!(history[:volfracs], volfracs)

            # Save iteration
            !isnothing(filename) && save_partial && @timeit "export" begin
                    filename_i = @sprintf "%s.%4.4d.vtu" filename mma.state.it
                    VTKGridFile(filename_i, grid) do vtk
                        write_cell_data(vtk, @view(mma.state.x[1:3:end]), "carbon")
                        write_cell_data(vtk, @view(mma.state.x[2:3:end]), "bamboo")
                        write_cell_data(vtk, @view(mma.state.x[3:3:end]), "theta")
                        pvd[mma.state.it] = vtk
                    end
                end

            # Apply continuation
            if inner_it >= 20 && Δf < 1e-4
                # if at max p, stop the optimization
                model.mat_interp.penal == 5.0 && break
                # else, increase p
                model.mat_interp.penal += 1.0
                @info "Updated p to $(model.mat_interp.penal)"
                inner_it = 0
            end
        end

        # Evaluate final design
        model.mat_interp.penal = 3.0
        fea!(results, mma.state.x)
        history[:final_compliance] = compliance(results, MMSOMP)
        @info "p = 3 equivalent compliance: $(round(history[:final_compliance], sigdigits=4))"
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        history[:final_compliance] = NaN
    end
    TimerOutputs.complement!(TopOpt.timer)
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

        qp_global, qp_material, qp_vonmises, qp_principalstress, qp_principaldir = stress(results)
        projector = L2Projector(ip^2, model.grid)

        # final result
        VTKGridFile("$(filename).final.vtu", grid) do vtk
            write_cell_data(vtk, @view(mma.state.x[1:3:end]), "carbon")
            write_cell_data(vtk, @view(mma.state.x[2:3:end]), "bamboo")
            write_cell_data(vtk, @view(mma.state.x[3:3:end]), "theta")
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
