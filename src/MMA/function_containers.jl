struct Objective{F,DF} <: Function
    obj::F
    dobj::DF
end

(obj::Objective)(x) = obj.obj(x), obj.dobj(x)

struct Constraints{VF<:Vector,VDF<:Vector} <: Function
    constraints::VF
    dconstraints::VDF
    m::Int
end
Constraints(g_vec::Vector, dg_vec::Vector) = Constraints(g_vec, dg_vec, length(g_vec))
Constraints(g, dg) = Constraints([g], [dg], 1)

(c::Constraints)(x) = [g(x) for g in c.constraints], reduce(vcat, [dg(x)' for dg in c.dconstraints])
