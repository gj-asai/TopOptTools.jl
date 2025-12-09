"""
Multi-material topology optimization for ecoefficiency
"""

using Ferrite, FerriteGmsh
using TimerOutputs, Printf, WriteVTK, HDF5, JLD2
using TopOptTools

mutable struct MMSOMP{dim,M,N,T<:Real,CT} <: MaterialInterpolation{N,T}
    mat::Vector{Material{dim,T,CT}}
    penal::T
end
MMSOMP(mat::Vector{Material{dim,T,CT}}, penal::T) where {dim,T,CT} = MMSOMP{dim,length(mat),length(mat) + 1,T,CT}(mat, penal)

function TopOptTools.interpolate(xe::AbstractVector{T}, interp::MMSOMP{2}) where {T<:Real}
    ρ, θ = xe[1:end-1], xe[end]
    result = zero(SymmetricTensor{4,2,T})
    for i in eachindex(interp.mat)
        weight = one(T)
        for j in eachindex(interp.mat)
            if i == j
                weight *= ρ[j]^interp.penal
            else
                weight *= one(T) - ρ[j]^interp.penal
            end
        end
        result += weight * rotate(interp.mat[i].C, θ)
    end
    result += TopOptTools.void(Val(2)).C
    return result
end

function mm_ecotopopt(comp_max, volfrac, rρ, rθ, angle=0; filename)
    reset_timer!()
    pvd = paraview_collection(filename)

    # read mesh
    mesh_file = "examples/models/mbb.msh" # this example contains a rectangular mesh
    grid = redirect_stdout(() -> togrid(mesh_file), devnull) # suppress print from the Gmsh.jl call inside togrid
    @info "Done reading $(mesh_file): $(getnnodes(grid)) nodes, $(getncells(grid)) elements"

    # loads and BCs
    xmax = maximum([n.x[1] for n in getnodes(grid)])
    ymax = maximum([n.x[2] for n in getnodes(grid)])
    addfacetset!(grid, "symmetry", x -> x[1] ≈ 0.0) # left edge
    addnodeset!(grid, "support", x -> x[1] ≈ xmax && x[2] ≈ 0.0) # bottom right corner
    addnodeset!(grid, "force", x -> x[1] ≈ 0.0 && x[2] ≈ ymax) # top left corner

    constraints = [
        Dirichlet(:u, getfacetset(grid, "symmetry"), (x, t) -> 0.0, [1]), # block x displacement
        Dirichlet(:u, getnodeset(grid, "support"), (x, t) -> 0.0, [2]), # block y displacement
    ]
    loads = [
        NodalLoad("force", (0.0, -100.0)),
    ]

    # materials
    carbon = Orthotropic2D(El=16.7e3, Et=3.7e3, nult=0.33, Glt=1.4e3, ρ=1.27e-3, CO2=10.1)
    bamboo = Orthotropic2D(El=10.8e3, Et=4.6e3, nult=0.36, Glt=1.7e3, ρ=1.12e-3, CO2=2.9)
    mat_interp = MMSOMP([carbon, bamboo], 1.0)

    # element interpolation and quadrature
    ip = Lagrange{RefQuadrilateral,1}() # linear elements
    qr = QuadratureRule{RefQuadrilateral}(2) # 2 point quadrature

    # FE model
    model = FEModel(; grid, ip, qr, mat_interp, constraints)
    f = compute_force_vector(loads, model)

    # initialize design variables
    x = collect(Iterators.flatten(zip(
        fill(volfrac, getncells(grid)),
        fill(volfrac, getncells(grid)),
        fill(deg2rad(angle), getncells(grid))
    )))
    xmin = collect(Iterators.flatten(zip(
        fill(0, getncells(grid)),
        fill(0, getncells(grid)),
        fill(-π, getncells(grid))
    )))
    xmax = collect(Iterators.flatten(zip(
        fill(1, getncells(grid)),
        fill(1, getncells(grid)),
        fill(π, getncells(grid))
    )))

    xTilde = copy(x)
    xPhys = copy(x)
    dPhysdTilde = similar(x)

    dCO2dx = similar(x)
    dcdx = similar(x)
    dvdx = similar(x)
    dgdx = zeros(2, 3 * getncells(grid))

    # initialize linear solver
    fesolver = FESolver(model)

    @timeit "build filters" begin
        @info "Building sensitivity filter with radius $rρ"
        density_filter = ConvolutionFilter(rρ, model)
        @info "Building orientation filter with radius $rθ"
        orientation_filter = ConvolutionFilter(rθ, model)
    end

    # initial impact
    @views CO2_ini =
        mat_interp.mat[1].CO2 * mat_interp.mat[1].ρ * xPhys[1:3:end] ⋅ model.elemvol +
        mat_interp.mat[2].CO2 * mat_interp.mat[2].ρ * xPhys[2:3:end] ⋅ model.elemvol

    history = Dict(
        :beta => Float64[],
        :penal => Float64[],
        :objective => Float64[],
        :constraint => Vector{Float64}[],
        :volfracs => Vector{Float64}[],
    )

    # initialize MMA
    m, n = 2, length(x)
    a0mma, amma, cmma, dmma = 1.0, zeros(m), fill(1000.0, m), zeros(m)
    mma = MMAWorkspace(m, n, a0mma, amma, cmma, dmma, asyinit=0.1, asyincr=1.1, asydecr=0.6, move=0.45)
    xold1, xold2 = similar(x), similar(x)

    beta = 1
    eta = 0.5
    @info "Starting optimization with beta = $(beta) and p = $(mat_interp.penal)"
    try
        maxiter = 1200
        continuation_iter = 0
        for loop in 1:maxiter
            continuation_iter += 1

            @timeit "filter and project" begin
                xTilde .= x

                @views TopOptTools.filter!(xTilde[1:3:end], density_filter)
                @views TopOptTools.filter!(xTilde[2:3:end], density_filter)
                @views TopOptTools.filter!(xTilde[3:3:end], orientation_filter)

                xPhys .= xTilde
                @views project_heaviside!(xPhys[1:3:end], beta, eta)
                @views project_heaviside!(xPhys[2:3:end], beta, eta)
            end

            @timeit "assemble stiffness" update_stiffness!(fesolver, xPhys)
            @timeit "linear solve" fea!(fesolver, f)

            @timeit "evaluate functions" begin
                # Objective: CO2 impact
                @views CO2 =
                    mat_interp.mat[1].CO2 * mat_interp.mat[1].ρ * xPhys[1:3:end] ⋅ model.elemvol +
                    mat_interp.mat[2].CO2 * mat_interp.mat[2].ρ * xPhys[2:3:end] ⋅ model.elemvol

                # Constraint: max compliance
                u = fesolver.solution
                c = dot(u, fesolver.K, u)

                # Constraint: max volume fraction
                @views v = xPhys[1:3:end] ⋅ model.elemvol + xPhys[2:3:end] ⋅ model.elemvol

                # all constraints
                g = [c / comp_max - 1, v / volfrac / sum(model.elemvol) - 1]
            end

            @timeit "sensitivity analysis" begin
                # Objective: CO2 impact
                dCO2dx[1:3:end] .= mat_interp.mat[1].CO2 * mat_interp.mat[1].ρ * model.elemvol
                dCO2dx[2:3:end] .= mat_interp.mat[2].CO2 * mat_interp.mat[2].ρ * model.elemvol
                dCO2dx[3:3:end] .= 0.0

                # Constraint: max compliance
                adjoint_sensitivities!(dcdx, u, u, fesolver)

                # Constraint: max volume fraction
                dvdx[1:3:end] .= model.elemvol
                dvdx[2:3:end] .= model.elemvol
                dvdx[3:3:end] .= 0.0

                # Chain rule
                dPhysdTilde .= xTilde
                @views project_heaviside_derivative!(dPhysdTilde, beta, eta)

                # densities
                for i in 1:2
                    dCO2dx[i:3:end] .*= dPhysdTilde[i:3:end] # xPhys -> xTilde
                    @views TopOptTools.filter!(dCO2dx[i:3:end], density_filter) # xTilde -> x

                    dcdx[i:3:end] .*= dPhysdTilde[i:3:end] # xPhys -> xTilde
                    @views TopOptTools.filter!(dcdx[i:3:end], density_filter) # xTilde -> x

                    dvdx[i:3:end] .*= dPhysdTilde[i:3:end] # xPhys -> xTilde
                    @views TopOptTools.filter!(dvdx[i:3:end], density_filter) # xTilde -> x
                end
                # angles, no need to filter dCO2dtheta and dvdtheta, they are all zero
                @views TopOptTools.filter!(dcdx[3:3:end], orientation_filter)

                # all constraints
                dgdx[1, :] .= dcdx / comp_max
                dgdx[2, :] .= dvdx / volfrac / sum(model.elemvol)
            end

            # Stopping criterion: relative change in the objective function
            change = if loop > 1
                abs(CO2 - history[:objective][end]) / history[:objective][end]
            else
                Inf
            end

            # log
            volfracs = [
                xPhys[1:3:end] ⋅ model.elemvol / sum(model.elemvol),
                xPhys[2:3:end] ⋅ model.elemvol / sum(model.elemvol),
            ]
            formatted_volfracs = join([@sprintf("%5.2f", 100 * f) for f in volfracs], ", ")
            @info @sprintf "It = %4d | CO2 = %8.4f | change = %8.2e | c = %9.4f | volfracs = [%s] %%" loop CO2 change c formatted_volfracs

            # update history
            push!(history[:beta], beta)
            push!(history[:penal], mat_interp.penal)
            push!(history[:objective], CO2)
            push!(history[:constraint], g)
            push!(history[:volfracs], volfracs)

            # export iteration file
            !isnothing(filename) && @timeit "export" begin
                filename_i = @sprintf "%s.%4.4d.vtu" filename (loop - 1)
                VTKGridFile(filename_i, grid) do vtk
                    write_cell_data(vtk, xPhys[1:3:end], "density1")
                    write_cell_data(vtk, xPhys[2:3:end], "density2")
                    write_cell_data(vtk, xPhys[3:3:end], "theta")
                    write_solution(vtk, model.dh, fesolver.solution)
                    pvd[loop] = vtk
                end
            end

            # apply continuation
            if (continuation_iter >= 100 && change < 1e-4) || continuation_iter >= 200
                # if at max beta, stop the optimization
                beta == 64 && break
                # else, increase beta and p, decrease gamma
                beta *= 2
                mat_interp.penal = min(3.0, mat_interp.penal + 1.0)
                @info "Updated beta to $(beta) and p to $(mat_interp.penal)"
                continuation_iter = 0
            end

            # MMA update
            @timeit "mma update" begin
                xnew = mma_update!(mma, continuation_iter,
                    x, xmin, xmax, xold1, xold2,
                    CO2 / CO2_ini, dCO2dx / CO2_ini, g, dgdx,
                )
                xold2 .= xold1
                xold1 .= x
                x .= xnew
            end
        end
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        # rethrow()
    finally
        !isnothing(filename) && @timeit "export" begin
            # history
            h5open(filename * ".h5", "w") do file
                write(file, "beta", history[:beta])
                write(file, "penal", history[:penal])
                write(file, "objective", history[:objective])
                write(file, "constraint", reduce(hcat, history[:constraint]))
                write(file, "volfracs", reduce(hcat, history[:volfracs]))
            end
            @info "Saved file $(filename).h5"

            # final design
            save("$(filename).design.jld2", "xPhys", xPhys)
            @info "Saved file $(filename).design.jld2"

            # paraview .pvd
            vtk_save(pvd)
            @info "Saved file $(filename).pvd"
        end

        print_timer(title="angle=$(angle)")
    end
end
