"""
Implementation of
G. Asai, C. Jansari, F. Lachaud, K. Masania and J. Morlier
Ecodesign of 3D volumetric fiber-composite structures with topology optimization
Composites: Part A (2025)
"""
# TODO: printability is not implemented

using Ferrite, Tensors, FerriteGmsh
using TimerOutputs, Printf, WriteVTK, HDF5, JLD2
using TopOpt

mutable struct mSOMP{dim,T<:Real,CT,V<:Vec} <: MaterialInterpolation{2,T}
    mat::Material{dim,T,CT}
    printing_direction::V
    penal::T
end
TopOpt.interpolate(xe::AbstractVector, somp::mSOMP) = xe[1]^somp.penal * rotate(somp.mat.C, somp.printing_direction, xe[2])

function bracket(volfrac, rρ, rθ, angle=0; filename)
    reset_timer!()
    pvd = paraview_collection(filename)

    # read mesh
    mesh_file = "examples/models/bracket.msh"
    grid = redirect_stdout(() -> togrid(mesh_file), devnull) # suppress print from the Gmsh.jl call inside togrid
    @info "Done reading $(mesh_file): $(getnnodes(grid)) nodes, $(getncells(grid)) elements"

    # loads and BCs
    constraints = [
        Dirichlet(:u, getfacetset(grid, "fixed"), (x, t) -> [0.0, 0.0, 0.0]),
    ]
    loads1 = [
        LinearLoad("force", (0.0, 0.0, 0.39)), # vertical, 355.86 N
    ]
    loads2 = [
        LinearLoad("force", (0.0, -0.414, 0.0)), # horizontal, 378.10 N
    ]

    # material
    flax_pla = Orthotropic3D(El=28.475e3, Et=6.48e3, nult=0.3725, Glt=1.17e3, ρ=1.38e-3, CO2=1.36)
    mat_interp = mSOMP(flax_pla, Vec{3}((0.0, 0.0, 1.0)), 1.0)

    # element interpolation and quadrature
    ip = Lagrange{RefTetrahedron,1}() # linear elements
    qr = QuadratureRule{RefTetrahedron}(2) # 2 point quadrature

    # FE model
    model = FEModel(; grid, ip, qr, mat_interp, constraints)
    f1 = compute_force_vector(loads1, model)
    f2 = compute_force_vector(loads2, model)

    # initialize design variables
    x = collect(Iterators.flatten(zip(
        fill(volfrac, getncells(grid)),
        fill(deg2rad(angle), getncells(grid))
    )))
    xmin = collect(Iterators.flatten(zip(
        fill(1e-3, getncells(grid)),
        fill(-π, getncells(grid))
    )))
    xmax = collect(Iterators.flatten(zip(
        fill(1, getncells(grid)),
        fill(π, getncells(grid))
    )))
    dcdx, dc1dx, dc2dx = similar(x), similar(x), similar(x)
    dgdx = zeros(1, 2 * getncells(grid))

    # passive regions
    for i in 1:getncells(grid)
        if sum((model.centers[1:2, i] - [0, 0]) .^ 2) < 10^2 ||
           sum((model.centers[1:2, i] - [0, -38]) .^ 2) < 10^2 ||
           sum((model.centers[1:2, i] - [-148, 18.5]) .^ 2) < 10^2 ||
           sum((model.centers[1:2, i] - [-148, -33.6]) .^ 2) < 10^2 ||
           sum((model.centers[2:3, i] - [-56.8, 44.7]) .^ 2) < 17.8^2
            x[2i-1] = 1
            xmin[2i-1] = 0.99
        end
    end

    # initialize FE solver
    fesolver = FESolver(x, model)

    @timeit "build filters" begin
        @info "Building sensitivity filter with radius $rρ"
        density_filter = ConvolutionFilter(rρ, model)
        @info "Building orientation filter with radius $rθ"
        orientation_filter = ConvolutionFilter(rθ, model)
    end

    history = Dict(
        :objective => Float64[],
        :constraint => Vector{Float64}[],
        :compliances => Vector{Float64}[],
    )

    # initialize MMA
    m, n = 1, length(x)
    a0mma, amma, cmma, dmma = 1.0, zeros(m), fill(1000.0, m), zeros(m)
    xold1, xold2 = similar(x), similar(x)
    mma = MMAWorkspace(m, n, a0mma, amma, cmma, dmma)

    @info "Starting optimization with p = $(mat_interp.penal)"
    try
        maxiter = 100
        max_grey = 0.3
        for loop in 1:maxiter+1
            @timeit "filtering" begin
                @views TopOpt.filter!(x[2:2:end], orientation_filter)
            end

            @timeit "assemble stiffness" update_stiffness!(fesolver, x)
            @timeit "linear solve" begin
                fea!(fesolver, f1)
                u1 = copy(fesolver.solution)

                fea!(fesolver, f2)
                u2 = copy(fesolver.solution)
            end

            @timeit "evaluate functions" begin
                # Objective: compliance
                c1 = dot(u1, fesolver.K, u1)
                c2 = dot(u2, fesolver.K, u2)
                c = (c1^8 + c2^8)^(1 / 8)

                # Constraint: max volume fraction
                v = x[1:2:end] ⋅ model.elemvol / sum(model.elemvol)
                @views g = [v / volfrac - 1,]
            end

            @timeit "sensitivity analysis" begin
                # Objective: compliance
                adjoint_sensitivities!(dc1dx, u1, u1, fesolver)
                adjoint_sensitivities!(dc2dx, u2, u2, fesolver)
                @views TopOpt.filter!(dc1dx[1:2:end], x[1:2:end], density_filter)
                @views TopOpt.filter!(dc2dx[1:2:end], x[1:2:end], density_filter)
                dcdx .= (c1 / c)^7 * dc1dx + (c2 / c)^7 * dc2dx

                # Constraint: max volume fraction
                dgdx[1, 1:2:end] .= model.elemvol / sum(model.elemvol) / volfrac
                dgdx[1, 2:2:end] .= 0.0
            end

            # Stopping criterion: relative change in the objective function
            change = if loop > 1
                abs(c - history[:objective][end]) / history[:objective][end]
            else
                Inf
            end
            @info @sprintf "It = %4d | c = %8.4f | c1 = %8.4f | c2 = %8.4f | volfrac = %6.4f | change = %8.2e" (loop - 1) c c1 c2 v change

            # update history
            push!(history[:objective], c)
            push!(history[:constraint], g)
            push!(history[:compliances], [c1, c2])

            # export iteration file
            !isnothing(filename) && @timeit "export" begin
                filename_i = @sprintf "%s.%4.4d.vtu" filename (loop - 1)
                VTKGridFile(filename_i, grid) do vtk
                    write_cell_data(vtk, x[1:2:end], "density")
                    write_cell_data(vtk, x[2:2:end], "theta")
                    write_solution(vtk, model.dh, u1, "1")
                    write_solution(vtk, model.dh, u2, "2")
                    pvd[loop] = vtk
                end
            end

            # apply continuation
            if change < 5e-3
                ρ = @view x[1:2:end]
                greyness = sum(0.1 .< ρ .< 0.9) / getncells(grid)
                @info @sprintf "Greyness is %4.1f %%" 100 * greyness

                mat_interp.penal >= 3 && greyness < max_grey && break
                mat_interp.penal += 1.0
                @info "Updated p to $(mat_interp.penal)"
            end

            # MMA update
            @timeit "mma update" begin
                xnew = mma_update!(mma, loop,
                    x, xmin, xmax, xold1, xold2,
                    c, dcdx, g, dgdx)
                xold2 .= xold1
                xold1 .= x
                x .= xnew
            end
        end
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        # rethrow()
    finally
        @timeit "export" begin
            # history
            h5open(filename * ".h5", "w") do file
                write(file, "objective", history[:objective])
                write(file, "constraint", reduce(hcat, history[:constraint]))
                write(file, "compliances", reduce(hcat, history[:compliances]))
            end
            @info "Saved file $(filename).h5"

            # final design
            save("$(filename).design.jld2", "x", x)
            @info "Saved file $(filename).design.jld2"

            # paraview .pvd
            vtk_save(pvd)
            @info "Saved file $(filename).pvd"
        end

        print_timer()
    end
end
