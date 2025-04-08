mutable struct SOMP{dim,T<:Real,CT} <: MaterialInterpolation{2,T}
    mat::Material{dim,T,CT}
    penal::T
end

function interpolate(xe::AbstractVector, interp::SOMP)
    ρ, θ = xe
    return ρ^interp.penal * rotate(interp.mat.C, θ)
end

function rotate_stress(global_stress::SymmetricTensor{2}, xe::AbstractVector, ::SOMP)
    θ = xe[2]
    return rotate(global_stress, -θ)
end

function compliance(_, results::FEResults{T}, ::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SOMP}
    @unpack K, u = results
    return u' * K * u
end

function dcompliance(_, results::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SOMP}
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

function impact(x, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SOMP}
    mat = model.mat_interp.mat
    ρ = @view x[1:2:end]
    return mat.CO2 * mat.ρ * ρ ⋅ model.elemvol
end

function dimpact(x, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SOMP}
    mat = model.mat_interp.mat
    dCO2dx = zero(x)
    dCO2dx[1:2:end] .= mat.CO2 * mat.ρ * model.elemvol
    return dCO2dx
end

function volume(x, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SOMP}
    ρ = @view x[1:2:end]
    return ρ ⋅ model.elemvol / sum(model.elemvol)
end

function dvolume(x, ::FEResults{T}, model::FEModel{dim,nvar,T,interp}) where {T,dim,nvar,interp<:SOMP}
    ∂g∂x = zero(x)
    ∂g∂x[1:2:end] .= model.elemvol / sum(model.elemvol)
    return ∂g∂x
end
