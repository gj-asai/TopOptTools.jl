struct LinearElasticity{nvar,MI<:MaterialInterpolation{nvar}} <: StiffnessBuilder{nvar}
    mat_interp::MI
end

function element_stiffness!(Ke::Matrix{T}, xe::AbstractVector, cellvalues::CellValues, stiff_builder::LinearElasticity) where {T<:Real}
    fill!(Ke, zero(T))

    # C is the same in all quadrature points
    C = interpolate(xe, stiff_builder.mat_interp)

    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            ∇sδεi = shape_symmetric_gradient(cellvalues, q_point, i)
            for j in 1:i
                ∇sδεj = shape_symmetric_gradient(cellvalues, q_point, j)
                Ke[i, j] += ∇sδεi ⊡ C ⊡ ∇sδεj * dΩ
            end
        end
    end

    for i in axes(Ke, 1), j in axes(Ke, 1)[begin+i:end]
        Ke[i, j] = Ke[j, i]
    end
end
