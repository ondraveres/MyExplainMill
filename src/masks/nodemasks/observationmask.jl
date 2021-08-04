"""
	`ObservationMask` is a mask that can be applied 
	to different type of nodes: Strings, Categorical Variables, Dense Arrays, etc.
"""
struct ObservationMask{M} <: AbstractListMask
	mask::M
end

Flux.@functor(ObservationMask)

ObservationMask(mask) = ObservationMask(mask, false)

function invalidate!(mk::ObservationMask, invalid_observations::AbstractVector{Int})
	invalidate!(mk.mask, invalid_observations)
end

function present(mk::ObservationMask, obs)
	map((&), obs, prunemask(mk.mask))
end

function foreach_mask(f, m::ObservationMask, level, visited)
	if !haskey(visited, m.mask)
		f(m.mask, level)
		visited[m.mask] = nothing
	end
end

function mapmask(f, m::ObservationMask, level, visited)
	new_mask = get!(visited, m.mask, f(m.mask, level))
	ObservationMask(new_mask)
end

