"""
MBB beam optimization using density filter

Follows the structure of
E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov and O. Sigmund
Efficient topology optimization in MATLAB using 88 lines of code
Structural and Multidisciplinary Optimization (2011)
"""
# INFO: plotting contains some workarounds needed until FerriteViz supports Ferrite v1.0

using Ferrite, FerriteGmsh, UnPack
using TimerOutputs, Printf, GLMakie
using TopOpt

struct SIMP{dim,T<:Real,CT} <: MaterialInterpolation{1,T}
    mat::Material{dim,T,CT}
    penal::T
end
TopOpt.interpolate(xe::AbstractVector, simp::SIMP) = xe[1]^simp.penal * simp.mat.C

function simp2(volfrac, rρ)
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
    f = compute_force_vector(loads, model)
    nelx, nely = [xmax, ymax] / sqrt(model.elemvol[1]) .|> round .|> Int # assuming identical square elements

    # initialize design variables
    x = DesignVector(1)
    foreach(1:getncells(model.grid)) do _
        push!(x, volfrac, 1e-3, 1)
    end
    xPhys = copy(x)

    # initialize FE solver
    fesolver = FESolver(x, model)

    @timeit "build filters" begin
        @info "Building density filter"
        density_filter = ConvolutionFilter(rρ, model)
    end

    # initialize MMA
    m, n = 1, length(x)
    a0mma, amma, cmma, dmma = 1.0, zeros(m), fill(1000.0, m), zeros(m)
    xold1, xold2 = similar(x), similar(x)
    mma = MMAWorkspace(m, n, a0mma, amma, cmma, dmma)

    @info "Starting optimization"
    try
        # plot initial design
        display(image(reshape(-xPhys.variables, nely, nelx)', interpolate=false, axis=(aspect=DataAspect(), yreversed=true, xreversed=true)))

        maxiter = 500
        for loop in 1:maxiter
            # FE analysis
            @timeit "assemble stiffness matrix" update_stiffness!(fesolver, xPhys)
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
                g = [xPhys ⋅ model.elemvol / sum(model.elemvol) / volfrac - 1,]
                dgdx = model.elemvol / sum(model.elemvol) / volfrac

                # chain rule
                TopOpt.filter!(dcdx, density_filter)
                TopOpt.filter!(dgdx, density_filter)
            end

            # log and plot
            change = norm(x - xold1, Inf)
            @info @sprintf "It = %4d | c = %10.4f | change = %8.2e" (loop - 1) c change
            empty!(current_axis())
            image!(reshape(-xPhys.variables, nely, nelx)', interpolate=false)

            # Stopping criterion: max change in the design variables
            change < 0.1 && break

            # MMA update
            @timeit "mma update" begin
                xnew = mma_update!(mma, m, n, loop,
                    x, x.lim_inf, x.lim_sup, xold1, xold2,
                    c, dcdx, g, dgdx')
                xold2 .= xold1
                xold1 .= x
                x .= xnew

                # filtering
                xPhys .= x
                TopOpt.filter!(xPhys, density_filter)
            end
        end
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        # rethrow()
    finally
        print_timer()
    end
end
