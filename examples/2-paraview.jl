"""
Same as example 1, but saves the results in Paraview files
- filename.h5: contains the convergence history
- filename.xxxx.vtu: Paraview file with results after intermediate iteration. Contains design variables, displacements and stresses
- filename.pvd: Paraview file to view iterations as time steps

Stress calculations are not efficient and are performed every iteration
"""

using Ferrite, FerriteGmsh
using TimerOutputs, Printf, WriteVTK, HDF5, JLD2
using TopOptTools

struct SIMP{dim,T<:Real,CT} <: MaterialInterpolation{1,T}
    mat::Material{dim,T,CT}
    penal::T
end
TopOptTools.interpolate(xe::AbstractVector, simp::SIMP) = xe[1]^simp.penal * simp.mat.C

# Returns stresses at the quadrature points for the result stored in solver
function stress(solver::FESolver)
    x = solver.x
    cellvalues = solver.model.cellvalues
    mat_interp = solver.model.mat_interp
    grid = solver.model.grid
    dh = solver.model.dh
    dim = TopOptTools.get_dim(solver.model)

    qp_global = [
        [zero(SymmetricTensor{2,dim}) for _ in 1:getnquadpoints(cellvalues)]
        for _ in 1:getncells(grid)
    ]
    qp_vonmises = [
        [zero(Float64) for _ in 1:getnquadpoints(cellvalues)]
        for _ in 1:getncells(grid)
    ]
    qp_principal = [
        [zero(SymmetricTensor{2,dim}) for _ in 1:getnquadpoints(cellvalues)]
        for _ in 1:getncells(grid)
    ]
    qp_directions = [
        [zeros(dim, dim) for _ in 1:getnquadpoints(cellvalues)]
        for _ in 1:getncells(grid)
    ]

    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues, cell)
        e = cellid(cell)
        cell_global = qp_global[e]
        cell_vonmises = qp_vonmises[e]
        cell_principal = qp_principal[e]
        cell_directions = qp_directions[e]
        for q_point in 1:getnquadpoints(cellvalues)
            xe = @views x[e:e]
            ε = function_symmetric_gradient(cellvalues, q_point, solver.solution, celldofs(cell))
            σ = TopOptTools.interpolate(xe, mat_interp) ⊡ ε
            s = dev(σ)

            cell_global[q_point] = σ
            cell_vonmises[q_point] = sqrt(1.5 * s ⊡ s)

            # sort eigenvalues by absolute value, largest first
            principal = eigen(σ)
            principal_order = sortperm(abs.(principal.values), rev=true)
            cell_principal[q_point] = diagm(SymmetricTensor{2,dim}, principal.values[principal_order])
            cell_directions[q_point] = principal.vectors[:, principal_order]
        end
    end

    return qp_global, qp_vonmises, qp_principal, qp_directions
end

function paraview(volfrac, rρ; filename)
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
    mat_interp = SIMP(mat, 3.0)

    # element interpolation and quadrature
    ip = Lagrange{RefQuadrilateral,1}() # linear elements
    qr = QuadratureRule{RefQuadrilateral}(2) # 2 point quadrature

    # FE model
    model = FEModel(; grid, ip, qr, mat_interp, constraints)
    f = compute_rhs(loads, model)

    # initialize design variables
    x = fill(volfrac, getncells(grid))
    xmin = fill(1e-3, getncells(grid))
    xmax = fill(1, getncells(grid))
    dcdx = similar(x)
    dgdx = zeros(1, getncells(grid))

    # initialize FE solver
    fesolver = FESolver(model)

    @timeit "build filters" begin
        @info "Building sensitivity filter with radius $rρ"
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
        maxiter = 1000
        for loop in 1:maxiter+1
            @timeit "assemble stiffness" update_stiffness!(fesolver, x)
            @timeit "linear solve" fea!(fesolver, f)

            @timeit "stresses" begin
                qp_global, qp_vonmises, qp_principalstress, qp_principaldir = stress(fesolver)
                projector = L2Projector(ip^2, grid)
            end

            @timeit "evaluate functions" begin
                # Objective: compliance
                u = fesolver.solution
                c = dot(u, fesolver.K, u)

                # Constraint: max volume fraction
                g = [x ⋅ model.elemvol / sum(model.elemvol) / volfrac - 1,]
            end

            @timeit "sensitivity analysis" begin
                # Objective: compliance
                adjoint_sensitivities!(dcdx, u, u, fesolver)

                # Constraint: max volume fraction
                dgdx .= model.elemvol' / sum(model.elemvol) / volfrac

                # sensitivity filtering
                TopOptTools.filter!(dcdx, x, density_filter)
            end

            # log
            change = norm(x - xold1, Inf)
            @info @sprintf "It = %4d | c = %10.4f | change = %5.3f" (loop - 1) c change

            # Update history
            push!(history[:objective], c)
            push!(history[:constraint], g)

            # Save iteration
            @timeit "export" begin
                filename_i = @sprintf "%s.%4.4d.vtu" filename (loop - 1)
                VTKGridFile(filename_i, grid) do vtk
                    @views write_cell_data(vtk, x, "density")
                    write_solution(vtk, model.dh, fesolver.solution)
                    write_projection(vtk, projector, project(projector, qp_global, qr), "stress - global")
                    write_projection(vtk, projector, project(projector, qp_vonmises, qr), "stress - von mises")
                    write_projection(vtk, projector, project(projector, qp_principalstress, qr), "stress - principal")

                    for i = 1:2
                        principaldir_i = [[Tensor{1,2}(vectors[:, i]) for vectors in el] for el in qp_principaldir]
                        write_projection(vtk, projector, project(projector, principaldir_i, qr), "principal direction $i")
                    end
                    pvd[loop] = vtk
                end
            end

            # Stopping criterion: max change in the design variables
            change < 0.1 && break

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
