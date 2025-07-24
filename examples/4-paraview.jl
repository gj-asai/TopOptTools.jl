"""
Same as example 3, but also saves result files if argument filename is not nothing
- filename.jld2: contains the convergence history
- filename.xxxx.vtu: Paraview file with results after intermediate iteration. Contains design variables, displacements and stresses
- filename.pvd: Paraview file to view iterations as time steps

Stress calculations are not optimized and are performed (relatively slowly) every iteration
"""
# INFO: plotting contains hardcoded workarounds needed until FerriteViz supports Ferrite v1.0
# TODO: plotting is missing the material orientations

using Ferrite, FerriteGmsh, UnPack
using TimerOutputs, Printf, WriteVTK, HDF5, GLMakie
using TopOpt

struct SOMP{dim,T<:Real,CT} <: MaterialInterpolation{2,T}
    mat::Material{dim,T,CT}
    penal::T
end
TopOpt.interpolate(xe::AbstractVector, somp::SOMP) = xe[1]^somp.penal * rotate(somp.mat.C, xe[2])

# Returns stresses at the quadrature points for the result stored in solver
function stress(solver::FESolver)
    x = solver.xPhys
    @unpack cellvalues, mat_interp, grid, dh = solver.model
    dim = TopOpt.get_dim(solver.model)

    qp_global = [
        [zero(SymmetricTensor{2,dim}) for _ in 1:getnquadpoints(cellvalues)]
        for _ in 1:getncells(grid)
    ]
    qp_material = [
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
        cell_material = qp_material[e]
        cell_vonmises = qp_vonmises[e]
        cell_principal = qp_principal[e]
        cell_directions = qp_directions[e]
        for q_point in 1:getnquadpoints(cellvalues)
            xe = @views x[2*e-1:2*e]
            ε = function_symmetric_gradient(cellvalues, q_point, solver.u, celldofs(cell))
            σ = TopOpt.interpolate(xe, mat_interp) ⊡ ε
            s = dev(σ)

            cell_global[q_point] = σ
            cell_material[q_point] = rotate(σ, -xe[2])
            cell_vonmises[q_point] = sqrt(1.5 * s ⊡ s)

            # sort eigenvalues by absolute value, largest first
            principal = eigen(σ)
            principal_order = sortperm(abs.(principal.values), rev=true)
            cell_principal[q_point] = diagm(SymmetricTensor{2,dim}, principal.values[principal_order])
            cell_directions[q_point] = principal.vectors[:, principal_order]
        end
    end

    return qp_global, qp_material, qp_vonmises, qp_principal, qp_directions
end

function somp_paraview(volfrac, rρ, rθ, angle=0; filename=nothing)
    reset_timer!()
    !isnothing(filename) && (pvd = paraview_collection(filename))

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

    # material
    mat = Orthotropic2D(El=10.48e3, Et=5.26e3, nult=0.39, Glt=1.89e3) # carbon fiber
    mat_interp = SOMP(mat, 3.0)

    # element interpolation and quadrature
    ip = Lagrange{RefQuadrilateral,1}() # linear elements
    qr = QuadratureRule{RefQuadrilateral}(2) # 2 point quadrature

    # FE model
    model = FEModel(; grid, ip, qr, mat_interp, constraints, loads)
    nelx, nely = [xmax, ymax] / sqrt(model.elemvol[1]) .|> round .|> Int # assuming identical square elements

    # initialize design variables
    x = DesignVector(2)
    foreach(1:getncells(grid)) do _
        push!(x, volfrac, 1e-3, 1)
        push!(x, deg2rad(angle), -π, π)
    end

    # initialize FE solver
    fesolver = FESolver(x, model)

    @timeit "build filters" begin
        @info "Building density filter"
        density_filter = ConvolutionFilter(rρ, model)
        @info "Building orientation filter"
        orientation_filter = ConvolutionFilter(rθ, model)
    end

    history = Dict(
        :objective => Float64[],
        :constraint => Vector{Float64}[],
    )

    # initialize MMA
    m, n = 1, length(x)
    a0mma, amma, cmma, dmma = 1.0, zeros(m), fill(1000.0, m), zeros(m)
    xold1, xold2 = similar(x), similar(x)
    mma = MMAWorkspace(m, n)

    @info "Starting optimization"
    try
        # plot initial design
        display(image(reshape(-x.variables[1:2:end], nely, nelx)', interpolate=false, axis=(aspect=DataAspect(), yreversed=true, xreversed=true)))

        maxiter = 1000
        for loop in 1:maxiter+1
            # FE analysis
            fea!(fesolver, x)

            @timeit "stresses" begin
                qp_global, qp_material, qp_vonmises, qp_principalstress, qp_principaldir = stress(fesolver)
                projector = L2Projector(ip^2, grid)
            end

            @timeit "sensitivity analysis" begin
                # Objective function: compliance
                @unpack u, K, ∂Ke∂x = fesolver
                c = dot(u, K, u)
                dcdx = zeros(length(∂Ke∂x))
                for cell in CellIterator(model.dh)
                    @views ue = u[celldofs(cell)]
                    e = cellid(cell)
                    dcdx[2*(e-1)+1] = -dot(ue, ∂Ke∂x[2*(e-1)+1], ue) # dcdrho
                    dcdx[2*(e-1)+2] = -dot(ue, ∂Ke∂x[2*(e-1)+2], ue) # dcdtheta
                end

                # Constraint: max volume fraction
                g = [x[1:2:end] ⋅ model.elemvol / sum(model.elemvol) / volfrac - 1,]
                dgdx = zeros(length(x))
                dgdx[1:2:end] .= model.elemvol / sum(model.elemvol) / volfrac

                # filtering
                @views TopOpt.filter!(dcdx[1:2:end], x[1:2:end], density_filter)
            end

            # log and plot
            change = norm((x-xold1)[1:2:end], Inf)
            @info @sprintf "It = %4d | c = %10.4f | change = %5.3f" (loop - 1) c change
            empty!(current_axis())
            @views image!(reshape(-x.variables[1:2:end], nely, nelx)', interpolate=false)

            # Update history
            push!(history[:objective], c)
            push!(history[:constraint], g)

            # Save iteration
            !isnothing(filename) && @timeit "export" begin
                filename_i = @sprintf "%s.%4.4d.vtu" filename (loop - 1)
                VTKGridFile(filename_i, grid) do vtk
                    @views write_cell_data(vtk, x[1:2:end], "density")
                    @views write_cell_data(vtk, x[2:2:end], "theta")
                    write_solution(vtk, model.dh, fesolver.u)
                    write_projection(vtk, projector, project(projector, qp_global, qr), "stress - global")
                    write_projection(vtk, projector, project(projector, qp_material, qr), "stress - material")
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
            change < 0.05 && break

            # MMA update
            @timeit "mma update" begin
                xnew = mma_update!(mma, m, n, loop,
                    x, x.lim_inf, x.lim_sup, xold1, xold2,
                    c, dcdx, g, dgdx',
                    a0mma, amma, cmma, dmma)
                xold2 .= xold1
                xold1 .= x
                x .= xnew

                # filtering
                @views TopOpt.filter!(x[2:2:end], orientation_filter)
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

            # paraview .pvd
            vtk_save(pvd)
            @info "Saved file $(filename).pvd"
        end

        # timing
        merge!(TimerOutputs.get_defaulttimer(), TopOpt.timer) # merge time measurements from the FE solver
        print_timer()
    end
end
