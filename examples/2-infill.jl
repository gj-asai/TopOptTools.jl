"""
Infill optimization

Implementation of
J. Wu, N. Aage, R. Westermann and O. Sigmund
Infill Optimization for Additive Manufacturing - Approaching Bone-like Porous Structures
IEEE Trans. on Visualization and Computer Graphics (2017)
"""
# WARNING: assumes that all elements are 1x1
# INFO: plotting contains some workarounds needed until FerriteViz supports Ferrite v1.0

using Ferrite, FerriteGmsh, UnPack
using TimerOutputs, Printf, GLMakie
using TopOpt

# mutable SIMP, accepts continuation on the penalization
mutable struct mSIMP{dim,T<:Real,CT} <: MaterialInterpolation{1,T}
    mat::Material{dim,T,CT}
    penal::T
end
TopOpt.interpolate(xe::AbstractVector, simp::mSIMP) = TopOpt.void(Val(2)).C + xe[1]^simp.penal * simp.mat.C

function infill(vol_local, rρ, rlocal)
    reset_timer!()

    # read mesh
    mesh_file = "examples/models/mbb_infill.msh" # this example contains a rectangular mesh
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
    model = FEModel(; grid, ip, qr, mat_interp, constraints, loads)
    nelx, nely = [xmax, ymax] / sqrt(model.elemvol[1]) .|> round .|> Int # assuming identical square elements

    # initialize design variables
    x = DesignVector(1)
    foreach(1:getncells(model.grid)) do _
        push!(x, vol_local, 1e-3, 1)
    end
    xTilde = copy(x)
    xPhys = copy(x)
    x_pde_hat = copy(x)

    # initialize FE solver
    fesolver = FESolver(xPhys, model)

    @timeit "build filters" begin
        @info "Building density filter"
        density_filter = ConvolutionFilter(rρ, model)
        @info "Build local volume filter"
        local_filter = PDEFilter(rlocal, model)
    end

    # initialize MMA
    m, n = 1, length(x)
    a0mma, amma, cmma, dmma = 1.0, zeros(m), fill(1000.0, m), zeros(m)
    xold1, xold2 = similar(x), similar(x)
    mma = MMAWorkspace(m, n, a0mma, amma, cmma, dmma, asyinit=0.5, asyincr=1.07, asydecr=0.65, move=0.01)

    beta = 1.0
    eta = 0.5
    plocal = 16
    vol_max_pnorm = (nelx * nely * vol_local^plocal)^(1 / plocal)

    @info "Starting optimization with beta = $beta, penal = $(mat_interp.penal)"
    try
        # plot initial design
        display(image(reshape(-xPhys.variables, nely, nelx)', interpolate=false, axis=(aspect=DataAspect(), yreversed=true, xreversed=true)))

        maxiter = 600
        loopbeta = 0
        for loop in 1:maxiter+1
            loopbeta += 1

            # FE analysis
            fea!(fesolver, xPhys)

            @timeit "sensitivity analysis" begin
                # Objective function: compliance
                @unpack u, K, ∂Ke∂x = fesolver
                c = dot(u, K, u)
                dcdx = zeros(length(∂Ke∂x))
                for cell in CellIterator(model.dh)
                    @views ue = u[celldofs(cell)]
                    dcdx[cellid(cell)] = -dot(ue, ∂Ke∂x[cellid(cell)], ue)
                end

                # Constraint: max local volume
                x_pde_hat .= xPhys
                TopOpt.filter!(x_pde_hat, local_filter)
                norm_xlocal = norm(x_pde_hat, plocal)
                g = [norm_xlocal - vol_max_pnorm,]
                dgdx = (x_pde_hat / norm_xlocal) .^ (plocal - 1)

                # chain rule
                dx = copy(xTilde)
                project_heaviside_derivative!(dx, beta, eta)

                dcdx .*= dx
                TopOpt.filter!(dcdx, density_filter)

                TopOpt.filter!(dgdx, local_filter)
                dgdx .*= dx
                TopOpt.filter!(dgdx, density_filter)
            end

            # log and plot
            change = norm(x - xold1, Inf)
            @info @sprintf "It = %4d | c = %10.4f | vol = %5.3f | cons = %8.4f" (loop - 1) c sum(xPhys) / sum(model.elemvol) g[1]
            empty!(current_axis())
            image!(reshape(-xPhys.variables, nely, nelx)', interpolate=false)

            # Continuation on beta and penal
            if beta < 100 && (loopbeta >= 40 || change < 1e-3)
                beta *= 2
                loopbeta = 0
                mat_interp.penal = min(3.0, mat_interp.penal + 0.5)
                @info "Updated beta to $(beta) and penal to $(mat_interp.penal)"
            end

            # MMA update
            @timeit "mma update" begin
                xnew = mma_update!(mma, m, n, loopbeta,
                    x, x.lim_inf, x.lim_sup, xold1, xold2,
                    c, dcdx, g, dgdx')
                xold2 .= xold1
                xold1 .= x
                x .= xnew

                # filtering
                xTilde .= x
                TopOpt.filter!(xTilde, density_filter)

                xPhys .= xTilde
                project_heaviside!(xPhys, beta, eta)
            end
        end
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        # rethrow()
    finally
        # timing
        merge!(TimerOutputs.get_defaulttimer(), TopOpt.timer) # merge time measurements from the FE solver
        print_timer()
    end
end
