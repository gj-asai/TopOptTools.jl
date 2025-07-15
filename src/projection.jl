"""
    project_heaviside!(x::AbstractVector, beta, threshold)

In-place smoothed Heaviside projection
"""
function project_heaviside!(x::AbstractVector, beta, threshold)
    @. x = (tanh(beta * threshold) + tanh(beta * (x - threshold))) / (tanh(beta * threshold) + tanh(beta * (1 - threshold)))
end

"""
    project_heaviside_derivative!(x::AbstractVector, beta, threshold)

Applies in-place the derivative of the smoothed Heaviside projection

The chain rule can be performed by multiplying the original vector by the result of this function
"""
function project_heaviside_derivative!(x::AbstractVector, beta, threshold)
    @. x = beta * sech(beta * (x - threshold))^2 / (tanh(beta * threshold) + tanh(beta * (1 - threshold)))
end
