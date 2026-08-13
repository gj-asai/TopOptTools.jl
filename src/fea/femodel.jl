"Struct that holds the geometric, material and boundary conditions data of a finite element model"
struct FEModel{dim,T<:Real,G<:Grid{dim},CV<:CellValues,FV<:FacetValues,IP<:Interpolation,QR<:QuadratureRule}
    grid::G
    centers::Matrix{T}
    elemvol::Vector{T}

    constraints::Vector{Dirichlet}

    cellvalues::CV
    facetvalues::FV
    ip::IP
    qr::QR

    dh::DofHandler{dim,G}
    ch::ConstraintHandler{DofHandler{dim,G},T}
end
get_dim(::FEModel{dim}) where {dim} = dim

"""
     FEModel(; grid::Grid, ip::Interpolation, qr::QuadratureRule, mat_interp::MaterialInterpolation, constraints::Vector{Dirichlet}) 

Creates the finite element model and also computes the element volumes and
a coloring of the grid for parallel assemble
"""
function FEModel(;
    grid::Grid{dim},
    ip::Interpolation,
    qr::QuadratureRule{shape},
    constraints::Vector{Dirichlet}
) where {dim,shape}
    # element type and quadrature rule
    cellvalues = CellValues(qr, ip^dim)

    qr_order = length(Ferrite.getpoints(qr))
    face_qr = FacetQuadratureRule{shape}(qr_order)
    facetvalues = FacetValues(face_qr, ip^dim)

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
        Ferrite.reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            elemvol[cellid(cell)] += getdetJdV(cellvalues, q_point)
        end
    end

    # element centroids
    centers = zeros(dim, getncells(grid))
    for cell in CellIterator(dh)
        id = cellid(cell)
        for node in getcoordinates(grid, id)
            centers[:, id] += node
        end
        centers[:, id] ./= Ferrite.nnodes_per_cell(grid, id)
    end

    return FEModel(grid, centers, elemvol, constraints, cellvalues, facetvalues, ip, qr, dh, ch)
end
