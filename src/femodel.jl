abstract type MaterialInterpolation{nvar} end

struct FEModel{dim,nvar,interp<:MaterialInterpolation{nvar}}
    grid::Grid{dim}
    elemvol::Vector{<:Real}

    mat_interp::interp
    constraints::Vector{Dirichlet}
    loads::Vector{<:Load{dim}}

    cellvalues::CellValues
    facetvalues::FacetValues
    dh::DofHandler
    ch::ConstraintHandler
    colors::Vector{Vector{Int}}
end

function FEModel(;
    grid::Grid{dim},
    ip::Interpolation,
    qr::QuadratureRule{shape},
    mat_interp::MaterialInterpolation,
    constraints::Vector{Dirichlet},
    loads::Vector{<:Load}
) where {dim,shape}
    # element type and quadrature rule
    cellvalues = CellValues(qr, ip^dim)

    qr_order = length(Ferrite.getpoints(qr))
    face_qr = FacetQuadratureRule{shape}(qr_order)
    facetvalues = FacetValues(face_qr, ip)
    colors = create_coloring(grid) # coloring for parallel assemble

    # degrees of freedom
    dh = DofHandler(grid)
    add!(dh, :u, ip^dim)
    close!(dh)

    # constraints
    ch = ConstraintHandler(dh)
    for cc in constraints
        add!(ch, cc)
    end
    close!(ch)

    # elemental volume vector
    elemvol = zeros(getncells(grid))
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            elemvol[cellid(cell)] += getdetJdV(cellvalues, q_point)
        end
    end

    return FEModel(grid, elemvol, mat_interp, constraints, loads, cellvalues, facetvalues, dh, ch, colors)
end

get_dim(::FEModel{dim,nvar,mat_interp}) where {dim,nvar,mat_interp} = dim
get_nvar(::FEModel{dim,nvar,mat_interp}) where {dim,nvar,mat_interp} = nvar

function get_centers(model::FEModel)
    centers = zeros(getncells(model.grid), get_dim(model))
    for cell in CellIterator(model.dh)
        id = cellid(cell)
        for node in getcoordinates(model.grid, id)
            centers[id, :] += node
        end
        centers[id, :] ./= Ferrite.nnodes_per_cell(model.grid, id)
    end
    return centers
end
