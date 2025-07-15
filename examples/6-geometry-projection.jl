"""
Implementation of
H. Smith, J. A. Norato
A MATLAB code for topology optimization using the geometry projection method
Structural and Multidisciplinary Optimization, 2020
"""

using Ferrite, FerriteGmsh, UnPack, LinearAlgebra
using TimerOutputs, Printf, WriteVTK, JLD2
using TopOpt

struct Linear{dim,T<:Real,CT} <: MaterialInterpolation{1,T}
    mat::Material{dim,T,CT}
end
TopOpt.interpolate(xe::AbstractVector, linear::Linear) = xe[1] * linear.mat.C

function geometry_projection(volfrac; filename=nothing)
    reset_timer!()
    !isnothing(filename) && (pvd = paraview_collection(filename))

    # read mesh
    mesh_file = "examples/models/mbb_gp.msh" # this example contains a rectangular mesh
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
        NodalLoad("force", (0.0, -0.1)),
    ]

    # material
    mat = Isotropic2D(E=1.0, nu=0.3)
    mat_interp = Linear(mat)

    # element interpolation and quadrature
    ip = Lagrange{RefQuadrilateral,1}() # linear elements
    qr = QuadratureRule{RefQuadrilateral}(2) # 2 point quadrature

    # FE model
    model = FEModel(; grid, ip, qr, mat_interp, constraints, loads)

    # initial end points of the bars
    points = [
        0.25 4.75; 4.75 4.75; 0.25 2.75; 4.75 4.75; 0.25 4.75; 4.75 2.75; 0.25 2.75; 4.75 2.75; 0.25 2.25; 4.75 2.25;
        0.25 0.25; 4.75 2.25; 0.25 2.25; 4.75 0.25; 0.25 0.25; 4.75 0.25; 5.25 4.75; 9.75 4.75; 5.25 2.75; 9.75 4.75;
        5.25 4.75; 9.75 2.75; 5.25 2.75; 9.75 2.75; 5.25 2.25; 9.75 2.25; 5.25 0.25; 9.75 2.25; 5.25 2.25; 9.75 0.25;
        5.25 0.25; 9.75 0.25; 10.25 4.75; 14.75 4.75; 10.25 2.75; 14.75 4.75; 10.25 4.75; 14.75 2.75; 10.25 2.75; 14.75 2.75;
        10.25 2.25; 14.75 2.25; 10.25 0.25; 14.75 2.25; 10.25 2.25; 14.75 0.25; 10.25 0.25; 14.75 0.25; 15.25 4.75; 19.75 4.75;
        15.25 2.75; 19.75 4.75; 15.25 4.75; 19.75 2.75; 15.25 2.75; 19.75 2.75; 15.25 2.25; 19.75 2.25; 15.25 0.25; 19.75 2.25;
        15.25 2.25; 19.75 0.25; 15.25 0.25; 19.75 0.25
    ]

    # initialize design variables
    z = DesignVector(6) # design variables: x1, y1, x2, y2, r, alpha
    for (x1, y1, x2, y2) in zip(points[1:2:end, 1], points[1:2:end, 2], points[2:2:end, 1], points[2:2:end, 2])
        push!(z, x1 / xmax, 0, 1)
        push!(z, y1 / ymax, 0, 1)
        push!(z, x2 / xmax, 0, 1)
        push!(z, y2 / ymax, 0, 1)
        push!(z, 0.25, 0.2499, 0.2501)
        push!(z, 0.5, 0, 1)
    end

    # preallocate densities
    nbars = length(z) ÷ 6
    num_elem = getncells(model.grid)
    dens_c = zeros(num_elem) # densities for compliance computation
    dens_v = zeros(num_elem) # densities for volume computation
    ddenscdz = zeros(length(dens_c), length(z))
    ddensvdz = zeros(length(dens_v), length(z))

    # Initialize linear solver
    fesolver = FESolver(dens_c, model)

    history = Dict(
        :objective => Float64[],
        :constraint => Vector{Float64}[],
    )

    # initialize mma
    m, n = 1, length(z)
    a0mma, amma, cmma, dmma = 1.0, zeros(m), fill(1000.0, m), zeros(m)
    zold1, zold2 = similar(z), similar(z)
    mma = MMAWorkspace(m, n; move=0.01)

    # geometry projection parameters
    ρmin = 1e-2
    p = 8 # smooth max norm
    q = 3 # SIMP
    h = sqrt(model.elemvol[1])
    r = h * sqrt(2) / 2

    @info "Starting optimization with $(nbars) bars"
    try
        maxiter = 300
        for loop in 1:maxiter+1
            @timeit "geometry projection" begin
                fill!(dens_c, 0.0)
                fill!(dens_v, 0.0)
                fill!(ddenscdz, 0.0)
                fill!(ddensvdz, 0.0)

                for b in 1:nbars
                    zb = z[6*(b-1)+1:6*b]
                    x1b = zb[1:2]
                    x2b = zb[3:4]
                    rb = zb[5]
                    αb = zb[6]

                    # rescaling points
                    domain_size = [xmax, ymax]
                    x1b .*= domain_size
                    x2b .*= domain_size

                    x21 = x2b - x1b
                    lb = norm(x21) # length of bar b
                    lb < 1e-12 && (lb = 1) # avoid division by zero
                    ab = x21 / lb # unit vector along the bar's medial axis

                    for e in 1:num_elem
                        xe = model.centers[:, e] # position of the centroid of the element

                        xe1 = xe - x1b
                        lbe = ab ⋅ xe1 # component of xe1 parallel to the bar
                        vec_rbe = xe1 - lbe * ab
                        rbe = norm(vec_rbe) # component of xe1 perpendicular to the bar

                        # distance from the the element to the medial segment of the bar
                        ∂dbe∂x = zeros(4)
                        if lbe <= 0
                            dbe = norm(xe1)
                            ∂dbe∂x[1:2] = -xe1 / dbe
                        elseif lbe > lb
                            xe2 = xe - x2b
                            dbe = norm(xe2)
                            ∂dbe∂x[3:4] = -xe2 / dbe
                        else
                            dbe = rbe
                            ∂dbe∂x[1:2] = -vec_rbe * (1 - lbe / lb) / dbe
                            ∂dbe∂x[3:4] = -vec_rbe * lbe / lb / dbe
                        end
                        dbe == 0 && fill!(∂dbe∂x, 0.0) # if point lies on the medial axis, sensitivity is zero
                        # because x variables are scaled:
                        ∂dbe∂x[1:2] .*= domain_size
                        ∂dbe∂x[3:4] .*= domain_size

                        # signed distance from the element to the boundary of the bar
                        ϕbe = rb - dbe
                        ∂ϕbe∂z = zeros(6)
                        ∂ϕbe∂z[1:4] = -∂dbe∂x # x1, y1, x2, y2
                        ∂ϕbe∂z[5] = 1.0 # rb
                        # αb sensitivity is zero

                        # density in element e due to bar b
                        xbe = ϕbe / r
                        ∂ρbe∂z = zeros(z.vars_per_element)
                        if xbe < -1
                            ρbe = 0
                        elseif xbe > 1
                            ρbe = 1
                        else
                            ρbe = 1 - (acos(xbe) + xbe * sqrt(1 - xbe^2)) / π
                            ∂ρbe∂z = 1 / r * 2 * sqrt(1 - xbe^2) / π * ∂ϕbe∂z
                        end

                        # apply size variable
                        ∂αb∂z = I[1:6, 6] # vector with all 0's except for position 6
                        ∂ρbe∂z = αb * ∂ρbe∂z + ρbe * ∂αb∂z
                        ρbe *= αb

                        # penalized densities
                        ρhatbe = ρbe^q
                        ∂ρhatbe∂z = q * ρbe^(q - 1) * ∂ρbe∂z

                        # add the contribution of this bar to the element densities
                        dens_c[e] += (1 - ρmin^p) * ρhatbe^p
                        dens_v[e] += (1 - ρmin^p) * ρbe^p
                        ddenscdz[e, 6*(b-1)+1:6*b] = (1 - ρmin^p) * ρhatbe^(p - 1) * ∂ρhatbe∂z
                        ddensvdz[e, 6*(b-1)+1:6*b] = (1 - ρmin^p) * ρbe^(p - 1) * ∂ρbe∂z
                    end
                end

                # after going through all bars, finish the calculation of the combined densities
                for e in 1:num_elem
                    # at this point, dens_c[e] contains (1-ρmin^p) Σ_b ρhatbe^p
                    dens_c[e] = (ρmin^p + dens_c[e])^(1 / p)
                    # at this point, ddenscdz[e,:] contains (1-ρmin^p) ρhatbe^(p-1) ∂ρhatbe∂z
                    ddenscdz[e, :] *= dens_c[e]^(1 - p)

                    # this is analogous
                    dens_v[e] = (ρmin^p + dens_v[e])^(1 / p)
                    ddensvdz[e, :] *= dens_v[e]^(1 - p)
                end
            end

            # FE analysis
            update_xPhys!(fesolver, dens_c)
            fea!(fesolver)

            @timeit "sensitivity analysis" begin
                # Objective: compliance
                @unpack u, K, ∂Ke∂x = fesolver
                c = dot(u, K, u)
                dcddensc = zeros(length(∂Ke∂x))
                for cell in CellIterator(model.dh)
                    @views ue = u[celldofs(cell)]
                    dcddensc[cellid(cell)] = -ue' * ∂Ke∂x[cellid(cell)] * ue
                end

                # Constraint: max volume fraction
                v = dens_v ⋅ model.elemvol
                dvddensv = copy(model.elemvol)
                g = [v / sum(model.elemvol) / volfrac - 1,]
                dgddensv = dvddensv / sum(model.elemvol) / volfrac

                # Chain rule
                dcdz = vec(dcddensc' * ddenscdz)
                dgdz = vec(dgddensv' * ddensvdz)
            end

            change = norm(z - zold1, Inf)
            @info @sprintf "It = %4d | c = %10.4f | cons = %7.4f | Δz = %8.2e " loop - 1 c g[1] change

            # update history
            push!(history[:objective], c)
            push!(history[:constraint], g)

            # save iteration
            !isnothing(filename) && @timeit "export" begin
                filename_i = @sprintf "%s.%4.4d.vtu" filename loop
                VTKGridFile(filename_i, grid) do vtk
                    write_cell_data(vtk, dens_c, "density")
                    write_solution(vtk, model.dh, fesolver.u)
                    pvd[loop] = vtk
                end
            end

            change < 1e-2 && break

            # MMA update
            @timeit "mma update" begin
                znew = mma_update!(mma, m, n, loop,
                    z, z.lim_inf, z.lim_sup, zold1, zold2,
                    c, dcdz, g, dgdz',
                    a0mma, amma, cmma, dmma
                )
                zold2 .= zold1
                zold1 .= z
                z .= znew
            end
        end
    catch e
        @warn "Computation interrupted - $(typeof(e))"
        # rethrow()
    finally
        !isnothing(filename) && @timeit "export" begin
            # history
            save("$(filename).jld2", "history", history)
            @info "Saved file $(filename).jld2"

            # paraview .pvd
            vtk_save(pvd)
            @info "Saved file $(filename).pvd"
        end

        # timing
        merge!(TimerOutputs.get_defaulttimer(), TopOpt.timer) # merge time measurements from the FE solver
        print_timer()
    end

end
