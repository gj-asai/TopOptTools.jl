"""
Type that defines how the design variables are interpolated for the element stiffness matrix

Every new optimization should define a subtype of `MaterialInterpolation` and the respective `interpolate(::AbstractVector, ::MaterialInterpolation)` method
"""
abstract type MaterialInterpolation{nvar,T<:Real} end
interpolate(::AbstractVector, interp::MaterialInterpolation) = throw("Method TopOpt.interpolate(::AbstractVector, ::$(typeof(interp))) is not defined")

