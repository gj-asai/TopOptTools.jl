"""
Same as example 1, but for the GE bracket in three dimensions
"""

using Ferrite, FerriteGmsh, UnPack
using TimerOutputs, Printf, WriteVTK, HDF5, JLD2
using TopOpt

struct SIMP{dim,T<:Real,CT} <: MaterialInterpolation{1,T}
    mat::Material{dim,T,CT}
    penal::T
end
TopOpt.interpolate(xe::AbstractVector, simp::SIMP) = xe[1]^simp.penal * simp.mat.C

function bracket(volfrac, rρ; filename=nothing)
    reset_timer!()
    !isnothing(filename) && (pvd = paraview_collection(filename))

    # read mesh
    mesh_file = "examples/models/bracket.msh" # this example contains a rectangular mesh
    grid = redirect_stdout(() -> togrid(mesh_file), devnull) # suppress print from the Gmsh.jl call inside togrid
    @info "Done reading $(mesh_file): $(getnnodes(grid)) nodes, $(getncells(grid)) elements"

    # loads and BCs
    constraints = [
        Dirichlet(:u, getfacetset(grid, "fixed"), (x, t) -> [0.0, 0.0, 0.0]),
    ]
    loads = [
        LinearLoad("force", (0.0, 0.0, 1.0)),
    ]

    # material
    mat = Isotropic3D(E=1e3, nu=0.3)
    mat_interp = SIMP(mat, 3.0)

    # element interpolation and quadrature
    ip = Lagrange{RefTetrahedron,1}() # linear elements
    qr = QuadratureRule{RefTetrahedron}(2) # 2 point quadrature

    # FE model
    model = FEModel(; grid, ip, qr, mat_interp, constraints)
    f = compute_force_vector(loads, model)

    # initialize design variables
    x = DesignVector(1)
    foreach(1:getncells(model.grid)) do _
        push!(x, volfrac, 1e-3, 1)
    end

    # initialize FE solver
    fesolver = FESolver(x, model)

    @timeit "build filters" begin
        @info "Building sensitivity filter"
        density_filter = ConvolutionFilter(rρ, model)
    end

    history = Dict(
        :objective => Float64[],
        :constraint => Vector{Float64}[],
    )

    # initialize MMA
    m, n = 1, length(x)
    a0mma, amma, cmma, dmma = 1.0, zeros(m), fill(1000.0, m), zeros(m)
    xold1, xold2 = similar(x), similar(x)
    mma = MMAWorkspace(m, n, a0mma, amma, cmma, dmma)

    @info "Starting optimization"
    try
        maxiter = 100
        for loop in 1:maxiter+1
            # FE analysis
            @timeit "assemble stiffness matrix" update_stiffness!(fesolver, x)
            @timeit "linear solve" fea!(fesolver, f)

            @timeit "sensitivity analysis" begin
                # Objective function: compliance
                @unpack u, K, ∂Ke∂x = fesolver
                c = dot(u, K, u)
                dcdx = zeros(length(∂Ke∂x))
                for cell in CellIterator(model.dh)
                    @views ue = u[celldofs(cell)]
                    dcdx[cellid(cell)] = -dot(ue, ∂Ke∂x[cellid(cell)], ue)
                end

                # Constraint: max volume fraction
                g = [x ⋅ model.elemvol / sum(model.elemvol) / volfrac - 1,]
                dgdx = model.elemvol' / sum(model.elemvol) / volfrac

                # filtering
                TopOpt.filter!(dcdx, x, density_filter)
            end

            # log and plot
            change = norm(x - xold1, Inf)
            @info @sprintf "It = %4d | c = %10.4f | change = %8.2e" (loop - 1) c change

            # update history
            push!(history[:objective], c)
            push!(history[:constraint], g)

            # export iteration file
            !isnothing(filename) && @timeit "export" begin
                filename_i = @sprintf "%s.%4.4d.vtu" filename (loop - 1)
                VTKGridFile(filename_i, grid) do vtk
                    write_cell_data(vtk, x, "density")
                    write_solution(vtk, model.dh, fesolver.u)
                    pvd[loop] = vtk
                end
            end

            # Stopping criterion: max change in the design variables
            change < 0.01 && break

            # MMA update
            @timeit "mma update" begin
                xnew = mma_update!(mma, m, n, loop,
                    x, x.lim_inf, x.lim_sup, xold1, xold2,
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
        !isnothing(filename) && @timeit "export" begin
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
