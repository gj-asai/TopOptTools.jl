struct SOMP{dim,T<:Real} <: MaterialInterpolation{2,T}
    mat::Material{dim,T}
    penal::T
end

function interpolate(xe::Vector, interp::SOMP)
    ρ, θ = xe
    return ρ^interp.penal * rotate(interp.mat.C, θ)
end

function compliance(_, results::FEResults, ::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SOMP}
    @unpack K, u = results
    return u' * K * u
end

function dcompliance(_, results::FEResults, model::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SOMP}
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

function impact(x, ::FEResults, model::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SOMP}
    mat = model.mat_interp.mat
    ρ = @view x[1:2:end]
    return mat.CO2 * mat.ρ * ρ ⋅ model.elemvol
end

function dimpact(_, ::FEResults, model::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SOMP}
    mat = model.mat_interp.mat
    return mat.CO2 * mat.ρ * model.elemvol
end

function volume(x, ::FEResults, model::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SOMP}
    ρ = @view x[1:2:end]
    return ρ ⋅ model.elemvol / sum(model.elemvol)
end

function dvolume(x, ::FEResults, model::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SOMP}
    ∂g∂x = zero(x)
    ∂g∂x[1:2:end] .= model.elemvol / sum(model.elemvol)
    return ∂g∂x
end
