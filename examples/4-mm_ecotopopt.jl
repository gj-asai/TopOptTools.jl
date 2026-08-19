"""
Multi-material topology optimization for ecoefficiency
"""

using Ferrite, FerriteGmsh
using TimerOutputs, Printf, GLMakie
using TopOptTools
import FerriteViz

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

function mm_ecotopopt(comp_max, volfrac, rρ, rθ, angle=0)
    reset_timer!()

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
    model = FEModel(; grid, ip, qr, constraints)
    f = compute_rhs(loads, model)

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
    fesolver = FEA(model, LinearElasticity(mat_interp))

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

    # history of the objective
    history = Float64[]

    # initialize MMA
    m, n = 2, length(x)
    a0mma, amma, cmma, dmma = 1.0, zeros(m), fill(1000.0, m), zeros(m)
    mma = MMAWorkspace(m, n, xmin, xmax, a0mma, amma, cmma, dmma, asyinit=0.1, asyincr=1.1, asydecr=0.6, move=0.5)

    beta = 1
    eta = 0.5
    @info "Starting optimization with beta = $(beta) and p = $(mat_interp.penal)"
    try
        # plot initial design
        fig = Figure()
        ax = Axis(fig[1, 1], aspect=DataAspect(), xautolimitmargin=(0, 0), yautolimitmargin=(0, 0), xgridvisible=false, ygridvisible=false)

        # colormaps with transparency for eadch material
        cmap1 = GLMakie.Colors.alphacolor.(resample_cmap(:Blues, 11), 0.0:0.1:1.0)
        cmap2 = GLMakie.Colors.alphacolor.(resample_cmap(:Reds, 11), 0.0:0.1:1.0)

        ds = FerriteViz.FEData(model.dh, fesolver.solution)
        FerriteViz.cellplot!(ax, ds, xPhys[1:3:end], colormap=cmap1)
        FerriteViz.cellplot!(ax, ds, xPhys[2:3:end], colormap=cmap2)
        display(fig)

        maxiter = 1200
        for loop in 1:maxiter
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
            @timeit "linear solve" solve!(fesolver, f)

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
                abs(CO2 - history[end]) / history[end]
            else
                Inf
            end
            push!(history, CO2)

            # log and plot
            volfracs = [
                xPhys[1:3:end] ⋅ model.elemvol / sum(model.elemvol),
                xPhys[2:3:end] ⋅ model.elemvol / sum(model.elemvol),
            ]
            formatted_volfracs = join([@sprintf("%5.2f", 100 * f) for f in volfracs], ", ")
            @info @sprintf "It = %4d | CO2 = %8.4f | change = %8.2e | c = %9.4f | volfracs = [%s] %%" loop CO2 change c formatted_volfracs
            empty!(ax)
            FerriteViz.cellplot!(ax, ds, xPhys[1:3:end], colormap=cmap1)
            FerriteViz.cellplot!(ax, ds, xPhys[2:3:end], colormap=cmap2)

            # apply continuation
            if (mma.iter >= 100 && change < 1e-4) || mma.iter >= 200
                # if at max beta, stop the optimization
                beta == 64 && break
                # else, increase beta and p, decrease gamma
                beta *= 2
                mat_interp.penal = min(3.0, mat_interp.penal + 1.0)
                @info "Updated beta to $(beta) and p to $(mat_interp.penal)"
                restart!(mma)
            end

            # MMA update
            @timeit "mma update" x .= mma_update!(mma, x, dCO2dx / CO2_ini, g, dgdx)
        end
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        # rethrow()
    finally
        print_timer(title="angle=$(angle)")
    end
end
