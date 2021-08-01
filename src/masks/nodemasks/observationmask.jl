"""
	`ObservationMask` is a mask that can be applied 
	to different type of nodes: Strings, Categorical Variables, Dense Arrays, etc.
"""
struct ObservationMask{M} <: AbstractListMask
	mask::M
end

Flux.@functor(ObservationMask)

function invalidate!(mk::ObservationMask, invalid_observations::AbstractVector{Int})
	invalidate!(mk.mask, invalid_observations)
end

function foreach_mask(f, m::ObservationMask, level = 1)
	f(m.mask, level)
end

function present(mk::ObservationMask, obs)
	map((&), obs, prunemask(mk.mask))
end

function mapmask(f, m::ObservationMask, level = 1)
	ObservationMask(f(m.mask, level))
end