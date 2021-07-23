struct SimpleMask{T,V<:AbstractVector{T}} <: AbstractVectorMask
	x::V
end

SimpleMask(d::Int) = SimpleMask(ones(Float32, d))
prunemask(m::SimpleMask{Bool,<:Any}) = m.x
prunemask(m::SimpleMask{<:Number,<:Any}) = m.x .> 0
diffmask(m::SimpleMask) = m.x
simplemask(m::SimpleMask) = m
Base.length(m::SimpleMask) = length(m.x)
Base.getindex(m::SimpleMask, i) = m.x[i]
Base.setindex!(m::SimpleMask, v, i) = m.x[i] = v
Base.materialize!(m::SimpleMask, v) = m.x .= v


struct HeuristicMask{T} <: AbstractVectorMask
	x::Vector{Bool}
	h::Vector{T}
end

HeuristicMask(d::Int) = HeuristicMask(zeros(Float32, d))
HeuristicMask(h::Vector) = HeuristicMask(fill(true, length(h)), h)

prunemask(m::HeuristicMask) = m.x .> 0.5
diffmask(m::HeuristicMask) = m.x
simplemask(m::HeuristicMask) = m
Base.length(m::HeuristicMask) = length(m.x)
Base.getindex(m::HeuristicMask, i) = m.x[i]
Base.setindex!(m::HeuristicMask, v, i) = m.x[i] = v
Base.materialize!(m::HeuristicMask, v) = m.x .= v
heuristic(m::HeuristicMask) = m.h
