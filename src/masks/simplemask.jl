struct SimpleMask{T} <: AbstractVectorMask
	x::Vector{T}
end

SimpleMask(d::Int) = SimpleMask(ones(Float32, d))
prunemask(m::SimpleMask{Bool}) = m.x
prunemask(m::SimpleMask{<:Number}) = m.x .> 0
diffmask(m::SimpleMask) = m.x
rawmask(m::SimpleMask) = m.x
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
rawmask(m::HeuristicMask) = m.x
Base.length(m::HeuristicMask) = length(m.x)
Base.getindex(m::HeuristicMask, i) = m.x[i]
Base.setindex!(m::HeuristicMask, v, i) = m.x[i] = v
Base.materialize!(m::HeuristicMask, v) = m.x .= v
heuristic(m::HeuristicMask) = m.h
