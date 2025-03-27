struct ConvolutionFilter{TH<:SparseMatrixCSC}
    H::TH
end

function ConvolutionFilter(radius::Float64, model::FEModel)
    if radius == 0
        n = getncells(model.grid)
        return ConvolutionFilter(sparse(I, n, n))
    end

    @info "Building convolution filter"
    centers = model.centers
    iH, jH = Int[], Int[]
    sH = Float64[]

    curi, curj = Int[], Int[]
    curs = Float64[]
    for i in 1:getncells(model.grid)
        empty!(curi)
        empty!(curj)
        empty!(curs)
        for j in inrange(model.balltree, centers[:, i], 1.2 * radius)
            dist = sqrt(sum((centers[:, i] - centers[:, j]) .^ 2))
            dist > radius && continue
            push!(curi, i)
            push!(curj, j)
            push!(curs, radius - dist)
        end
        curs ./= sum(curs) # normalize row
        append!(iH, curi)
        append!(jH, curj)
        append!(sH, curs)
    end

    H = sparse(iH, jH, sH)
    return ConvolutionFilter(H)
end

function filter!(x::AbstractVector, f::ConvolutionFilter)
    x .= f.H * x
end

function filter!(x::AbstractVector, weight::AbstractVector, f::ConvolutionFilter)
    x .= f.H * (weight .* x) ./ weight
end
