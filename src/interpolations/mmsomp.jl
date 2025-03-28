mutable struct MMSOMP{dim,T<:Real,M,N,CT} <: MaterialInterpolation{N,T}
    mat::Vector{Material{dim,T,CT}}
    penal::T
end
MMSOMP(mat::Vector{Material{dim,T,CT}}, penal::T) where {dim,T,CT} = MMSOMP{dim,T,length(mat),length(mat) + 1,CT}(mat, penal)

function interpolate(xe::AbstractVector{T}, interp::MMSOMP{dim}) where {T<:Real,dim}
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
    return result
end
rotate_mmsomp(mat::Material{2}, θ::T) where {T<:Real} = rotate(mat.C, θ)
rotate_mmsomp(mat::Material{3}, θ::T) where {T<:Real} = rotate(mat.C, Vec{3}((0.,0.,1.)), θ)

function compliance(_, results::FEResults{T}, ::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:MMSOMP}
    @unpack K, u = results
    return u' * K * u
end

function dcompliance(_, results::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:MMSOMP}
    @unpack u, ∂Ke∂x = results
    dcdx = zeros(length(∂Ke∂x))
    for cell in CellIterator(model.dh)
        ue = u[celldofs(cell)]
        e = cellid(cell)
        for i in 1:nvar
            dcdx[nvar*(e-1)+i] = -ue' * ∂Ke∂x[nvar*(e-1)+i] * ue
        end
    end
    return dcdx
end

function impact(x, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:MMSOMP}
    CO2 = 0
    for (i, mati) in enumerate(model.mat_interp.mat)
        ρi = x[i:nvar:end]
        CO2 += mati.CO2 * mati.ρ * ρi ⋅ model.elemvol
    end
    return CO2
end

function dimpact(x, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:MMSOMP}
    dCO2dx = zero(x)
    for (i, mati) in enumerate(model.mat_interp.mat)
        dCO2dx[i:nvar:end] .= mati.CO2 * mati.ρ * model.elemvol
    end
    return dCO2dx
end

function volume(x, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:MMSOMP}
    ρsum = [sum(x[nvar*(i-1)+1:nvar*i-1]) for i in 1:getncells(model.grid)]
    return ρsum ⋅ model.elemvol / sum(model.elemvol)
end

function dvolume(x, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:MMSOMP}
    ∂g∂x = zero(x)
    for i = 1:length(model.mat_interp.mat)
        ∂g∂x[i:nvar:end] .= model.elemvol / sum(model.elemvol)
    end
    return ∂g∂x
end
