struct SIMP <: MaterialInterpolation{1}
    mat::Material
    penal::Real
end

function interpolate(xe::Vector, interp::SIMP)
    ρ = xe[1]
    return ρ^interp.penal * interp.mat.C
end

function compliance(_, results::FEResults, ::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SIMP}
    @unpack K, u = results
    return u' * K * u
end

function dcompliance(_, results::FEResults, model::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SIMP}
    @unpack u, ∂Ke∂x = results
    dcdx = zeros(length(∂Ke∂x))
    for cell in CellIterator(model.dh)
        ue = @view u[celldofs(cell)]
        dcdx[cellid(cell)] = -ue' * ∂Ke∂x[cellid(cell)] * ue
    end
    return dcdx
end

function impact(x, ::FEResults, model::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SIMP}
    mat = model.mat_interp.mat
    return mat.CO2 * mat.ρ * x ⋅ model.elemvol
end

function dimpact(_, ::FEResults, model::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SIMP}
    mat = model.mat_interp.mat
    return mat.CO2 * mat.ρ * model.elemvol
end

function volume(x, ::FEResults, model::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SIMP}
    return x ⋅ model.elemvol / sum(model.elemvol)
end

function dvolume(_, ::FEResults, model::FEModel{dim,nvar,interp}) where {dim,nvar,interp<:SIMP}
    return model.elemvol / sum(model.elemvol)
end
