struct SimpleMask{T,V<:AbstractVector{T}} <: AbstractVectorMask
	x::V
end

SimpleMask{T} = SimpleMask{T,V} where {V<:AbstractVector{T}}
SimpleMask(d::Int) = SimpleMask(ones(Float32, d))
prunemask(m::SimpleMask{Bool}) = m.x
prunemask(m::SimpleMask{<:Number}) = m.x .> 0
diffmask(m::SimpleMask) = m.x
simplemask(m::SimpleMask) = m
Base.length(m::SimpleMask) = length(m.x)
Base.getindex(m::SimpleMask, i) = m.x[i]
Base.setindex!(m::SimpleMask, v, i) = m.x[i] = v
Base.materialize!(m::SimpleMask, v) = m.x .= v
heuristic(m::SimpleMask) = fill(0, length(m.x))

"""
	HeuristicMask

	we do not assume it to be differentiable
"""
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

struct FollowingMasks{T<:Tuple} <: AbstractVectorMask
	masks::T

	#assert that all masks have the same length
	function FollowingMasks(masks::T) where {T<:Tuple}
		isempty(masks) && error("FollowingMasks cannot be empty")
		l = length(masks[1])
		!all(l == length(x) for x in masks) && error("All masks in FollowingMasks has to have same length")
		new{T}(masks)
	end
end

prunemask(m::FollowingMasks) = mapfoldl(prunemask, (a, b) -> max.(a, b), m.masks)
diffmask(m::FollowingMasks) = mapfoldl(diffmask, (a, b) -> max.(a, b), m.masks)
Base.length(m::FollowingMasks) = length(m.masks[1])
Base.getindex(m::FollowingMasks, i) = maximum(x[i] for x in m.masks)
support_participation(m::FollowingMasks) = true
invalidate!(m::FollowingMasks, i) = nothing
