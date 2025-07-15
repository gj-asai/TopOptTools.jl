"""
Multi-material topology optimization for ecoefficiency

Using a three field representation for the densities:
    - Design variables ∈ [0,1] are filtered using a convolution filter of radius rρ
    - Filtered field is projected using a smooth Heaviside function, with continuation on its sharpness

Material orientation is represented by one angle ∈ [-π,π]
    - Filtered using a convolution filter of radius rθ

The interpolated constitutive matrix is given by:
C = C_void + Σ_i ( xi^p Π_(j≠i) (1 - xj^p) * C_0,i ),
where:
C_0,i are the constitutive matrices of each candidate material
Cvoid is the constitutive matrix of an isotropic material with very low stiffness
p is similar to SIMP penalization, to which a continuation is applied starting from 1 and increasing to 3 in steps of 1

Objective function: wimpact*CO2 + (1-wimpact)*compliance, with wimpact ∈ (0,1)
CO2 and compliance are normalized against the average values of the structures completely made of each of the candidate materials

Constraint: max volume fraction, including all materials
"""

using Ferrite, FerriteGmsh, UnPack
using TimerOutputs, Printf, WriteVTK, JLD2
using TopOpt

mutable struct MMSOMP{dim,M,N,T<:Real,CT} <: MaterialInterpolation{N,T}
    mat::Vector{Material{dim,T,CT}}
    penal::T
end
MMSOMP(mat::Vector{Material{dim,T,CT}}, penal::T) where {dim,T,CT} = MMSOMP{dim,length(mat),length(mat) + 1,T,CT}(mat, penal)

function TopOpt.interpolate(xe::AbstractVector{T}, interp::MMSOMP{dim}) where {T<:Real,dim}
    ρ, θ = xe[1:end-1], xe[end]
    result = zero(SymmetricTensor{4,dim,T})
    for i in eachindex(interp.mat)
        weight = one(T)
        for j in eachindex(interp.mat)
            if i == j
                weight *= ρ[j]^interp.penal
            else
                weight *= one(T) - ρ[j]^interp.penal
            end
        end
        result += weight * rotate_mmsomp(interp.mat[i], θ)
    end
    result += TopOpt.void(Val(dim)).C
    return result
end
rotate_mmsomp(mat::Material{2}, θ) = rotate(mat.C, θ)
rotate_mmsomp(mat::Material{3}, θ) = rotate(mat.C, Vec{3}((0.0, 0.0, 1.0)), θ)

function mm_ecotopopt1(volfrac, rρ, rθ, wimpact, angle=0; filename=nothing)
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

    # materials
    carbon = Orthotropic2D(El=122.98e3, Et=10.2e3, nult=0.25, Glt=3.67e3, ρ=1.54e-3, CO2=11.29)
    bamboo = Orthotropic2D(El=10.48e3, Et=5.26e3, nult=0.39, Glt=1.89e3, ρ=0.98e-3, CO2=1.668)
    mat_interp = MMSOMP([carbon, bamboo], 1.0)

    # element interpolation and quadrature
    ip = Lagrange{RefQuadrilateral,1}() # linear elements
    qr = QuadratureRule{RefQuadrilateral}(2) # 2 point quadrature

    # FE model
    model = FEModel(; grid, ip, qr, mat_interp, constraints, loads)

    # initialize design variables
    nmaterials = length(mat_interp.mat)
    x = DesignVector(nmaterials + 1)
    foreach(1:getncells(grid)) do _
        foreach(1:nmaterials) do _
            push!(x, volfrac / nmaterials, 0, 1)
        end
        push!(x, 0.0, -π, π) # angles are initialized at 0 for the calculation of the normalization factors
    end
    xTilde = copy(x)
    xPhys = copy(x)

    # initialize linear solver
    fesolver = FESolver(xPhys, model)

    @timeit "build filters" begin
        @info "Building density filter"
        density_filter = ConvolutionFilter(rρ, model)
        @info "Building orientation filter"
        orientation_filter = ConvolutionFilter(rθ, model)
    end

    # normalization factors for the objective function
    fea!(fesolver)
    @unpack K, u = fesolver

    # average of the compliances of the completely solid structures made of each material
    # just multiplying by volfrac because p = 1 at this point
    comp_ini = volfrac * dot(u, K, u)

    # initial impact
    CO2_ini = 0
    for i in 1:nmaterials
        mati = mat_interp.mat[i]
        ρi = x[i:nmaterials+1:end]
        CO2_ini += mati.CO2 * mati.ρ * ρi ⋅ model.elemvol
    end
    @info @sprintf "Normalization factors: compliance => %.4f, CO2 => %.4f" comp_ini CO2_ini

    # now setting the correct initial orientation
    x[nmaterials+1:nmaterials+1:end] .= deg2rad(angle)
    xTilde .= x
    xPhys .= x

    history = Dict(
        :compliance => Float64[],
        :impact => Float64[],
        :beta => Float64[],
        :penal => Float64[],
        :objective => Float64[],
        :constraint => Vector{Float64}[],
        :volfracs => Vector{Float64}[],
    )

    # initialize MMA
    m, n = 1, length(x)
    mma = MMAWorkspace(m, n, asyinit=0.1, asyincr=1.1, asydecr=0.5, move=0.5)
    a0mma, amma, cmma, dmma = 1.0, zeros(m), fill(1000.0, m), zeros(m)
    xold1, xold2 = similar(x), similar(x)

    beta = 1
    eta = 0.5
    @info "Starting optimization with beta = $(beta) and p = $(mat_interp.penal)"
    try
        maxiter = 1000
        continuation_iter = 0
        for loop in 1:maxiter+1
            continuation_iter += 1

            # FE analysis
            update_xPhys!(fesolver, xPhys)
            fea!(fesolver)

            @timeit "sensitivity analysis" begin
                # Compliance
                @unpack u, K, ∂Ke∂x = fesolver
                c = dot(u, K, u)
                dcdx = zeros(length(∂Ke∂x))
                for cell in CellIterator(model.dh)
                    @views ue = u[celldofs(cell)]
                    e = cellid(cell)
                    nvar = nmaterials + 1
                    for i in 1:nvar
                        dcdx[nvar*(e-1)+i] = -dot(ue, ∂Ke∂x[nvar*(e-1)+i], ue)
                    end
                end

                # CO2 impact
                CO2 = 0
                dCO2dx = zeros(length(x))
                for i in 1:nmaterials
                    mati = mat_interp.mat[i]
                    @views ρi = xPhys[i:nmaterials+1:end]
                    CO2 += mati.CO2 * mati.ρ * ρi ⋅ model.elemvol
                    dCO2dx[i:nmaterials+1:end] .= mati.CO2 * mati.ρ * model.elemvol
                end

                # Objective function: wimpact*CO2 + (1-wimpact)*compliance
                obj = wimpact * CO2 / CO2_ini + (1 - wimpact) * c / comp_ini
                dobjdx = wimpact * dCO2dx / CO2_ini + (1 - wimpact) * dcdx / comp_ini

                # Constraint: max volume fraction
                v = 0
                dvdx = zeros(length(x))
                for i in 1:nmaterials
                    @views v += xPhys[i:nmaterials+1:end] ⋅ model.elemvol
                    dvdx[i:nmaterials+1:end] = model.elemvol
                end
                g = [v / sum(model.elemvol) / volfrac - 1,]
                dgdx = dvdx / sum(model.elemvol) / volfrac

                # chain rule
                dPhysdTilde = copy(xTilde)
                project_heaviside_derivative!(dPhysdTilde, beta, eta)
                # densities
                for i in 1:nmaterials
                    dobjdx[i:nmaterials+1:end] .*= dPhysdTilde[i:nmaterials+1:end] # xPhys -> xTilde
                    @views TopOpt.filter!(dobjdx[i:nmaterials+1:end], density_filter) # xTilde -> x

                    dgdx[i:nmaterials+1:end] .*= dPhysdTilde[i:nmaterials+1:end] # xPhys -> xTilde
                    @views TopOpt.filter!(dgdx[i:nmaterials+1:end], density_filter) # xTilde -> x
                end
                # angles, no need to filter dgdtheta, they are all zero
                @views TopOpt.filter!(dobjdx[nmaterials+1:nmaterials+1:end], orientation_filter)
            end

            # volume fractions of each material
            volfracs = Float64[]
            for i = 1:nmaterials
                push!(volfracs, xPhys[i:nmaterials+1:end] ⋅ model.elemvol / sum(model.elemvol))
            end

            # Stopping criterion: relative change in the objective function
            change = if loop > 1
                abs(obj - history[:objective][end]) / history[:objective][end]
            else
                Inf
            end

            formatted_volfracs = join([@sprintf("%5.2f", 100 * f) for f in volfracs], ", ")
            @info @sprintf "It = %4d | obj = %8.4f | Δobj = %8.2e | c = %10.4f | CO2 = %8.4f | volfracs = [%s] %%" loop obj change c CO2 formatted_volfracs

            # update history
            push!(history[:compliance], c)
            push!(history[:impact], CO2)
            push!(history[:beta], beta)
            push!(history[:penal], mat_interp.penal)
            push!(history[:objective], obj)
            push!(history[:constraint], g)
            push!(history[:volfracs], volfracs)

            # export iteration file
            !isnothing(filename) && @timeit "export" begin
                filename_i = @sprintf "%s.%4.4d.vtu" filename (loop - 1)
                VTKGridFile(filename_i, grid) do vtk
                    for i in 1:nmaterials
                        write_cell_data(vtk, xPhys[i:nmaterials+1:end], "density$(i)")
                    end
                    write_cell_data(vtk, xPhys[nmaterials+1:nmaterials+1:end], "theta")
                    write_solution(vtk, model.dh, fesolver.u)
                    pvd[loop] = vtk
                end
            end

            # apply continuation
            if continuation_iter >= 20 && change < 1e-4 || continuation_iter >= 100
                # if at max beta, stop the optimization
                beta == 64 && break
                # else, increase beta and p
                beta *= 2
                mat_interp.penal = min(3.0, mat_interp.penal + 1.0)
                @info "Updated beta to $(beta) and p to $(mat_interp.penal)"
                continuation_iter = 0
            end

            # MMA update
            @timeit "mma update" begin
                xnew = mma_update!(mma, m, n, loop,
                    x, x.lim_inf, x.lim_sup, xold1, xold2,
                    obj, dobjdx, g, dgdx',
                    a0mma, amma, cmma, dmma)
                xold2 .= xold1
                xold1 .= x
                x .= xnew

                # filtering
                xTilde .= x
                for i in 1:nmaterials
                    @views TopOpt.filter!(xTilde[i:nmaterials+1:end], density_filter)
                end
                @views TopOpt.filter!(xTilde[nmaterials+1:nmaterials+1:end], orientation_filter)

                # projection
                xPhys .= xTilde
                for i in 1:nmaterials
                    @views project_heaviside!(xPhys[i:nmaterials+1:end], beta, eta)
                end
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
        print_timer(title="w=$(wimpact), angle=$(angle)")
    end
end
