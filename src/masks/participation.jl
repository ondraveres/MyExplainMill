"""
	struct ParticipationTracker{T,M<:AbstractVectorMask} <: AbstractVectorMask
		m::M
		x::Vector{T}
	end

	The `participation` allows to track, if a part of a mask has an effect of the output of a classifier on a given sample. For example imagine that we have a following sample 
	```
	ds = BagNode(
		ArrayNode(
			NGramMatrix(["a","b","c","d","e"])
			)
		[1:5]
		)
	```

	`BagMask` is masking individual observations and so does `NGramMatrix`. 
	This means that if `BagMask` has mask `[true,false,true,false,true]`, 
	then explanation of `NGramMatrix` can consider only `["a","c","e"]` instead of 
	all items, because `["b","d"]` are removed by the `BagMask` of above. 
	The `participation(mk::NGramMatrixMask)` should therefore 
	return `[true,false,true,false,true]`.

	`ParticipationTracker` is a decorator of `<:AbstractVectorMask`. It would reexport 
	api of the `<:AbstractVectorMask`, therefore it would be invisible to it and also 
	it would allow to override it for the clustering.  
"""

struct ParticipationTracker{M<:AbstractVectorMask} <: AbstractVectorMask
	m::M
	p::Vector{Bool}
end

ParticipationTracker(m::AbstractVectorMask) = support_participation(m) ? m : ParticipationTracker(m, fill(true,length(m)))

prunemask(m::ParticipationTracker) = prunemask(m.m)
diffmask(m::ParticipationTracker) = diffmask(m.m)
simplemask(m::ParticipationTracker) = simplemask(m.m)
heuristic(m::ParticipationTracker) = heuristic(m.m)
Base.length(m::ParticipationTracker) = length(m.m)
Base.getindex(m::ParticipationTracker, args...) = getindex(m.m, args...)
Base.setindex!(m::ParticipationTracker, args...) = setindex!(m.m, args...)
Base.materialize!(m::ParticipationTracker, b::Base.Broadcast.Broadcasted) = Base.materialize!(m.m, b)

participate(m::ParticipationTracker) = m.p
support_participation(m::ParticipationTracker) = true
invalidate!(m::ParticipationTracker, i::Int) = m.p[i] = false
invalidate!(m::ParticipationTracker, i) = m.p[i] .= false
