struct ConvolutionFilter{TH<:SparseMatrixCSC}
    H::TH
end

"""
    ConvolutionFilter(radius::Float64, model::FEModel)

Computes the weights of a convolution filter of radius `radius` for the mesh stored in `model`
"""
function ConvolutionFilter(radius::Float64, model::FEModel)
    n = getncells(model.grid)

    radius == 0 && return ConvolutionFilter(sparse(I, n, n))

    centers = model.centers
    iH, jH = Int[], Int[]
    sH = Float64[]

    tmp_s = Float64[]
    idx_cols = Int[]
    for i in 1:n
        empty!(tmp_s)
        empty!(idx_cols)

        @views inrange!(idx_cols, model.balltree, centers[:, i], radius)
        for j in idx_cols
            dist = @views norm(centers[:, i] - centers[:, j])
            push!(iH, i)
            push!(jH, j)
            push!(tmp_s, radius - dist)
        end

        # normalize row
        tmp_s ./= sum(tmp_s)
        append!(sH, tmp_s)
    end

    H = sparse(iH, jH, sH)
    return ConvolutionFilter(H)
end

"""
    filter!(x::AbstractVector, f::ConvolutionFilter)

In-place filtering of `x`
"""
function filter!(x::AbstractVector, f::ConvolutionFilter)
    x .= f.H * x
end

"""
    filter!(x::AbstractVector, weight::AbstractVector, f::ConvolutionFilter)

In-place filtering of `x` using weights `weight`:
weight * xfiltered = H * (weight * x)
"""
function filter!(x::AbstractVector, weight::AbstractVector, f::ConvolutionFilter)
    x .*= weight
    filter!(x, f)
    x ./= weight
end
