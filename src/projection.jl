function project_heaviside!(x::AbstractVector, beta, threshold)
    @. x = (tanh(beta * threshold) + tanh(beta * (x - threshold))) / (tanh(beta * threshold) + tanh(beta * (1 - threshold)))
end

function project_heaviside_derivative!(x::AbstractVector, beta, threshold)
    @. x = beta * sech(beta * (x - threshold))^2 / (tanh(beta * threshold) + tanh(beta * (1 - threshold)))
end
