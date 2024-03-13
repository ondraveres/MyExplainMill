function entropy(x::AbstractArray)
    x = x[:]
    @assert all((x .>= 0) .& (x .<= 1))
    mask = (x .!= 0) .& (x .!= 1)
    -sum(x[mask] .* log.(x[mask])) / max(1, length(x))
end

_entropy(x::T) where {T<:Real} = (x <= 0 || x >= 1) ? zero(T) : -x * log(x)
function _entropy(x::AbstractArray)
    isempty(x) && return (0.0f0)
    mapreduce(_entropy, +, x) / length(x)
end

_∇entropy(Δ, x::T) where {T<:Real} = (x <= 0 || x >= 1) ? zero(T) : -Δ * (one(T) + log(x))

function _∇entropy(Δ, x)
    Δ /= length(x)
    (_∇entropy.(Δ, x),)
end

Zygote.@adjoint function entropy(x::AbstractArray)
    @assert all((x .>= 0) .& (x .<= 1))
    return (_entropy(x),
        Δ -> _∇entropy(Δ, x))
end
