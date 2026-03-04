"""
Implementation of
J. Wu, N. Aage, R. Westermann and O. Sigmund
Infill Optimization for Additive Manufacturing - Approaching Bone-like Porous Structures
IEEE Trans. on Visualization and Computer Graphics (2017)
"""

using Ferrite, FerriteGmsh
using TimerOutputs, Printf, WriteVTK, HDF5, JLD2
using TopOptTools

# mutable SIMP, accepts continuation on the penalization
mutable struct mSIMP{dim,T<:Real,CT} <: MaterialInterpolation{1,T}
    mat::Material{dim,T,CT}
    penal::T
end
TopOptTools.interpolate(xe::AbstractVector, simp::mSIMP) = TopOptTools.void(Val(2)).C + xe[1]^simp.penal * simp.mat.C

function infill(vol_local, rρ, rlocal; filename)
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
        NodalLoad("force", (0.0, -1.0)),
    ]

    # material
    mat = Isotropic2D(E=1.0, nu=0.3)
    mat_interp = mSIMP(mat, 1.0)

    # element interpolation and quadrature
    ip = Lagrange{RefQuadrilateral,1}() # linear elements
    qr = QuadratureRule{RefQuadrilateral}(2) # 2 point quadrature

    # FE model
    model = FEModel(; grid, ip, qr, mat_interp, constraints)
    f = compute_rhs(loads, model)

    # initialize design variables
    x = fill(vol_local, getncells(grid))
    xmin = fill(0, getncells(grid))
    xmax = fill(1, getncells(grid))
    dcdx = similar(x)
    dgdx = similar(x)

    xTilde = copy(x)
    xPhys = copy(x)
    x_pde_hat = copy(x)

    # initialize FE solver
    fesolver = FESolver(model)

    @timeit "build filters" begin
        @info "Building density filter with radius $rρ"
        density_filter = ConvolutionFilter(rρ, model)
        @info "Build local volume filter with radius $rlocal"
        local_filter = PDEFilter(rlocal, model)
    end

    history = Dict(
        :objective => Float64[],
        :constraint => Vector{Float64}[],
        :beta => Float64[],
        :penal => Float64[],
    )

    # initialize MMA
    m, n = 1, length(x)
    a0mma, amma, cmma, dmma = 1.0, zeros(m), fill(1000.0, m), zeros(m)
    mma = MMAWorkspace(m, n, xmin, xmax, a0mma, amma, cmma, dmma, asyinit=0.5, asyincr=1.07, asydecr=0.65, move=0.01)

    beta = 1.0
    eta = 0.5
    plocal = 16
    vol_max_pnorm = (sum(model.elemvol) * vol_local^plocal)^(1 / plocal)

    @info "Starting optimization with beta = $beta, penal = $(mat_interp.penal)"
    try
        maxiter = 600
        for loop in 1:maxiter+1
            @timeit "filter and project" begin
                xTilde .= x
                TopOptTools.filter!(xTilde, density_filter)

                xPhys .= xTilde
                project_heaviside!(xPhys, beta, eta)
            end

            @timeit "assemble stiffness" update_stiffness!(fesolver, xPhys)
            @timeit "linear solve" fea!(fesolver, f)

            @timeit "evaluate functions" begin
                # Objective: compliance
                u = fesolver.solution
                c = dot(u, fesolver.K, u)

                # Constraint: max local volume
                x_pde_hat .= xPhys
                TopOptTools.filter!(x_pde_hat, local_filter)
                norm_xlocal = norm(x_pde_hat, plocal)
                g = [norm_xlocal - vol_max_pnorm,]
            end

            @timeit "sensitivity analysis" begin
                # Objective: compliance
                adjoint_sensitivities!(dcdx, u, u, fesolver)

                # Constraint: max local volume
                dgdx .= (x_pde_hat / norm_xlocal) .^ (plocal - 1)

                # chain rule
                dx = copy(xTilde)
                project_heaviside_derivative!(dx, beta, eta)

                dcdx .*= dx
                TopOptTools.filter!(dcdx, density_filter)

                TopOptTools.filter!(dgdx, local_filter)
                dgdx .*= dx
                TopOptTools.filter!(dgdx, density_filter)
            end

            # log
            change = norm(x - mma.xold1, Inf)
            @info @sprintf "It = %4d | c = %10.4f | vol = %5.3f | cons = %8.4f" (loop - 1) c xPhys ⋅ model.elemvol / sum(model.elemvol) g[1]

            # Update history
            push!(history[:objective], c)
            push!(history[:constraint], g)
            push!(history[:beta], beta)
            push!(history[:penal], mat_interp.penal)

            # Save iteration
            @timeit "export" begin
                filename_i = @sprintf "%s.%4.4d.vtu" filename (loop - 1)
                VTKGridFile(filename_i, grid) do vtk
                    @views write_cell_data(vtk, xPhys, "density")
                    write_solution(vtk, model.dh, fesolver.solution)
                    pvd[loop] = vtk
                end
            end

            # Continuation on beta and penal
            if beta < 100 && (mma.iter >= 40 || change < 1e-3)
                restart!(mma)
                beta *= 2
                mat_interp.penal = min(3.0, mat_interp.penal + 0.5)
                @info "Updated beta to $(beta) and penal to $(mat_interp.penal)"
            end

            # MMA update
            @timeit "mma update" x .= mma_update!(mma, x, dcdx, g, dgdx')
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
