"""
Implementation of the linear thermal actuator from
O. Sigmund
Design of multiphysics actuators using topology optimization - Part I: One-material structures
Computer Methods in Applied Mechanics and Engineering (2000)
"""

using Ferrite, FerriteGmsh
using TimerOutputs, Printf, GLMakie
using TopOptTools
import FerriteViz

# defining a power law MaterialInterpolation with 1 design variable per element
struct SIMP{dim,T<:Real,CT} <: MaterialInterpolation{1,T}
    mat::Material{dim,T,CT}
    penal::T
end
TopOptTools.interpolate(xe::AbstractVector, simp::SIMP) = xe[1]^simp.penal * simp.mat.C

function thermal_actuator(volfrac, rρ)
    reset_timer!()

    # read mesh
    mesh_file = "examples/models/mbb.msh" # this example contains a rectangular mesh
    grid = redirect_stdout(() -> togrid(mesh_file), devnull) # suppress print from the Gmsh.jl call inside togrid
    @info "Done reading $(mesh_file): $(getnnodes(grid)) nodes, $(getncells(grid)) elements"

    # loads and BCs
    xmax = maximum([n.x[1] for n in getnodes(grid)])
    ymax = maximum([n.x[2] for n in getnodes(grid)])
    addfacetset!(grid, "left", x -> x[1] ≈ 0.0) # left edge
    addfacetset!(grid, "top", x -> x[2] ≈ ymax) # top edge
    addfacetset!(grid, "bottom", x -> x[2] ≈ 0.0) # bottom edge
    addnodeset!(grid, "output", x -> x[1] ≈ xmax && x[2] ≈ 0.0) # bottom right corner

    constraints = [
        Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0]), # fixed
        Dirichlet(:u, getfacetset(grid, "top"), (x, t) -> [0.0, 0.0]), # fixed
        Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, [2]), # symmetry
    ]

    # material
    mat = Isotropic2D(E=200e3, nu=0.31) # MPa, -
    mat_interp = SIMP(mat, 3.0)
    α = 15e-6 * one(Tensor{2,2}) # K^(-1)
    ΔT = 100 # K
    kspring = 10 # N/mm

    loads = [
        EigenstrainLoad(xe -> TopOptTools.interpolate(xe, mat_interp) ⊡ α * ΔT, dim=2),
    ]

    # element interpolation and quadrature
    ip = Lagrange{RefQuadrilateral,1}() # linear elements
    qr = QuadratureRule{RefQuadrilateral}(2) # 2 point quadrature

    # FE model
    model = FEModel(; grid, ip, qr, mat_interp, constraints)

    # initialize design variables
    x = fill(volfrac, getncells(grid))
    xmin = fill(1e-3, getncells(grid))
    xmax = fill(1, getncells(grid))
    dudx = similar(x)
    dgdx = zeros(1, getncells(grid))

    # initialize FE solver
    fesolver = FESolver(model)
    f = zeros(ndofs(model.dh))
    ∂f∂x = zeros(ndofs(model.dh), length(x))

    # objective function = L ⋅ u = x displacement at node "output"
    L = compute_rhs(
        [NodalLoad("output", (-1.0, 0.0))], # flip sign so we maximize it
        model
    )
    output_dof = findfirst(L .!= 0) # we will manually put a spring on this dof

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
    mma = MMAWorkspace(m, n, xmin, xmax, a0mma, amma, cmma, dmma, move=0.05)

    @info "Starting optimization"
    try
        # plot initial design
        fig = Figure()
        ax = Axis(fig[1, 1], aspect=DataAspect(), xautolimitmargin=(0, 0), yautolimitmargin=(0, 0))

        plotter = FerriteViz.MakiePlotter(model.dh, fesolver.solution)
        FerriteViz.cellplot!(ax, plotter, x, colormap=:binary)
        display(fig)

        maxiter = 500
        for loop in 1:maxiter+1
            @timeit "assemble stiffness" begin
                update_stiffness!(fesolver, x)
                fesolver.K[output_dof, output_dof] += kspring
            end

            @timeit "assemble load vector" compute_rhs!(f, ∂f∂x, loads, model, x)
            @timeit "linear solve" fea!(fesolver, f)

            @timeit "evaluate functions" begin
                # Objective: compliance
                u = copy(fesolver.solution)
                u_out = L ⋅ u

                # Constraint: max volume fraction
                g = [x ⋅ model.elemvol / sum(model.elemvol) / volfrac - 1,]
            end

            @timeit "sensitivity analysis" begin
                # adjoint analysis
                fea!(fesolver, L)
                lambda = fesolver.solution
                adjoint_sensitivities!(dudx, lambda, u, fesolver)
                for cell in CellIterator(model.dh)
                    e = cellid(cell)
                    dudx[e] += dot(lambda, ∂f∂x[:, e])
                end

                # Constraint: max volume fraction
                dgdx .= model.elemvol' / sum(model.elemvol) / volfrac

                # sensitivity filtering
                TopOptTools.filter!(dudx, x, density_filter)
            end

            # Stopping criterion: relative change in the objective function
            change = if loop > 1
                abs((u_out - history[:objective][end]) / history[:objective][end])
            else
                Inf
            end

            # log and plot
            @info @sprintf "It = %4d | u_out = %7.4f | change = %8.2e" (loop - 1) -u_out change
            empty!(ax)
            FerriteViz.cellplot!(ax, plotter, x, colormap=:binary)

            # Update history
            push!(history[:objective], u_out)
            push!(history[:constraint], g)

            # Stopping criterion: max change in the objecvtive function
            change < 1e-3 && break

            @timeit "mma update" x .= mma_update!(mma, x, dudx, g, dgdx)
        end
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        # rethrow()
    finally
        print_timer()
    end
end
