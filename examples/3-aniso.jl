"""
Anisotropic MBB beam topology optimization
Design variables are one density ∈ [0,1] and one angle ∈ [-π, π] for each element
Convolution filters are applied on the density sensitivities (radius rρ) and angles (radius rθ)
Stopping criterion is the max change in density
"""
# INFO: plotting contains hardcoded workarounds needed until FerriteViz supports Ferrite v1.0
# TODO: plotting is missing the material orientations

using Ferrite, FerriteGmsh, UnPack
using TimerOutputs, Printf, GLMakie
using TopOpt

struct SOMP{dim,T<:Real,CT} <: MaterialInterpolation{2,T}
    mat::Material{dim,T,CT}
    penal::T
end
TopOpt.interpolate(xe::AbstractVector, somp::SOMP) = xe[1]^somp.penal * rotate(somp.mat.C, xe[2])

function somp(volfrac, rρ, rθ, angle=0)
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

    # material
    mat = Orthotropic2D(El=10.48e3, Et=5.26e3, nult=0.39, Glt=1.89e3) # carbon fiber
    mat_interp = SOMP(mat, 3.0)

    # element interpolation and quadrature
    ip = Lagrange{RefQuadrilateral,1}() # linear elements
    qr = QuadratureRule{RefQuadrilateral}(2) # 2 point quadrature

    # FE model
    model = FEModel(; grid, ip, qr, mat_interp, constraints)
    f = compute_force_vector(loads, model)
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

    # initialize MMA
    m, n = 1, length(x)
    a0mma, amma, cmma, dmma = 1.0, zeros(m), fill(1000.0, m), zeros(m)
    xold1, xold2 = similar(x), similar(x)
    mma = MMAWorkspace(m, n, a0mma, amma, cmma, dmma)

    @info "Starting optimization"
    try
        # plot initial design
        @views display(image(reshape(-x.variables[1:2:end], nely, nelx)', interpolate=false, axis=(aspect=DataAspect(), yreversed=true, xreversed=true)))

        maxiter = 1000
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
                    ue = u[celldofs(cell)]
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

            # Stopping criterion: max change in density
            change < 0.02 && break

            # MMA update
            @timeit "mma update" begin
                xnew = mma_update!(mma, m, n, loop,
                    x, x.lim_inf, x.lim_sup, xold1, xold2,
                    c, dcdx, g, dgdx')
                xold2 .= xold1
                xold1 .= x
                x .= xnew

                @views TopOpt.filter!(x[2:2:end], orientation_filter)
            end
        end
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        # rethrow()
    finally
        print_timer()
    end
end
