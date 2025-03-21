struct ConvolutionFilter{T<:Real}
    H::Union{SparseMatrixCSC{T},LinearAlgebra.UniformScaling{T}}
end

function ConvolutionFilter(radius::Float64, model::FEModel)
    radius == 0 && return ConvolutionFilter{Float64}(I)

    @info "Building convolution filter"
    centers = get_centers(model) |> transpose
    tree = BallTree(centers)

    iH, jH = Int[], Int[]
    sH = Float64[]
    for i in 1:getncells(model.grid), j in inrange(tree, centers[:, i], 1.5 * radius)
        dist = sqrt(sum((centers[:, i] - centers[:, j]) .^ 2))
        dist > radius && continue
        push!(iH, i)
        push!(jH, j)
        push!(sH, radius - dist)
    end

    H = sparse(iH, jH, sH)
    H ./= sum(H, dims=2)
    dropzeros!(H)

    return ConvolutionFilter{Float64}(H)
end

function filter!(x::AbstractVector, f::ConvolutionFilter)
    x .= f.H * x
end

function filter!(x::AbstractVector, weight::AbstractVector, f::ConvolutionFilter)
    x .= f.H * (weight .* x) ./ weight
end
