abstract type MaterialInterpolation{nvar,T<:Real} end

struct FEModel{dim,nvar,T<:Real,interp<:MaterialInterpolation{nvar,T},G<:Grid{dim},BT<:BallTree,CV<:CellValues,FV<:FacetValues}
    grid::G
    balltree::BT
    centers::Matrix{T}
    elemvol::Vector{T}

    mat_interp::interp
    constraints::Vector{Dirichlet}
    loads::Vector{<:Load{dim,T}}

    cellvalues::CV
    facetvalues::FV
    dh::DofHandler{dim,G}
    ch::ConstraintHandler{DofHandler{dim,G},T}
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

    centers = zeros(dim, getncells(grid))
    for cell in CellIterator(dh)
        id = cellid(cell)
        for node in getcoordinates(grid, id)
            centers[:, id] += node
        end
        centers[:, id] ./= Ferrite.nnodes_per_cell(grid, id)
    end
    tree = BallTree(centers)

    return FEModel(grid, tree, centers, elemvol, mat_interp, constraints, loads, cellvalues, facetvalues, dh, ch, colors)
end

get_dim(::FEModel{dim}) where {dim} = dim
get_nvar(::FEModel{dim,nvar}) where {dim,nvar} = nvar
