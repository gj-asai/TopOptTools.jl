struct SIMP{dim,T<:Real} <: MaterialInterpolation{1,T}
    mat::Material{dim,T}
    penal::T
end

function interpolate(xe::Vector, interp::SIMP)
    ρ = xe[1]
    return ρ^interp.penal * interp.mat.C
end

function compliance(_, results::FEResults{T}, ::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SIMP}
    @unpack K, u = results
    return u' * K * u
end

function dcompliance(_, results::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SIMP}
    @unpack u, ∂Ke∂x = results
    dcdx = zeros(length(∂Ke∂x))
    for cell in CellIterator(model.dh)
        ue = @view u[celldofs(cell)]
        dcdx[cellid(cell)] = -ue' * ∂Ke∂x[cellid(cell)] * ue
    end
    return dcdx
end

function impact(x, ::FEResults{T}, model::FEModel{dim,nvar,interp}) where {T,dim,nvar,interp<:SIMP}
    mat = model.mat_interp.mat
    return mat.CO2 * mat.ρ * x ⋅ model.elemvol
end

function dimpact(_, ::FEResults{T}, model::FEModel{dim,nvar,interp}) where {T,dim,nvar,interp<:SIMP}
    mat = model.mat_interp.mat
    return mat.CO2 * mat.ρ * model.elemvol
end

function volume(x, ::FEResults{T}, model::FEModel{dim,nvar,interp}) where {T,dim,nvar,interp<:SIMP}
    return x ⋅ model.elemvol / sum(model.elemvol)
end

function dvolume(_, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SIMP}
    return model.elemvol / sum(model.elemvol)
end
