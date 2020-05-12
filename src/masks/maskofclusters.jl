"""
struct MaskOfClusters{I<:Union{Nothing, Vector{Int}}}
	mask::Array{Bool,1}
	participate::Array{Bool,1}
	outputid::Array{Int,1}
	stats::D
	cluster_membership::I
end

holds the structure for explaining 
mask --- a binary indicator vector, where each item corresponds to one explainable feature in the sample. 
	It is used to prune or modulate the sample. 

participate --- identifies if the corresponding item in the mask participate in the creation of modulated samples. 
	For example imagine to have two bags, one nested in the other. If we remove instance(s) in the top bag, then this 
	correnspods to whole bags in the bottom mil problem. Now whatever values the mask of these (removed bag) has, 
	it does not have any effect on the sample, as they are effectively removed. For these samples we set participate
	to zero, such that these items will not be counted in the statistics.

outputid --- this is a vector which identifies to which sample the corresponding item belongs. This is used to 
	speeedup the update of stats when multiple samples are used

daf --- Shappley value statistics for each item (cluster of items)

cluster_membership --- this identifies to which cluster the item belongs to. 
	This is created if clustering of items is on
"""
struct MaskOfClusters{D}
	mask::Array{Bool,1}
	participate::Array{Bool,1}
	outputid::Array{Int,1}
	stats::D
	cluster_membership::Vector{Int}
end

participate(m::MaskOfClusters) = m.participate
mask(m::MaskOfClusters) = m.mask

Base.length(m::MaskOfClusters) = length(m.stats)
Base.getindex(m::MaskOfClusters{Vector{Int}}, i::Int) = m.mask[m.cluster_membership .== i]
Base.setindex!(m::MaskOfClusters{Vector{Int}}, v, i::Int) = m.mask[m.cluster_membership .== i] .= v
Base.fill!(m::MaskOfClusters, v) = Base.fill!(prunemask(m), v)

####
#	Explaination, where items are clustered together
####
function MaskOfClusters(cluster_membership::Vector{T}, initstats) where {T<:Integer} 
	cluster_membership = normalize_clusterids(cluster_membership)
	n = length(unique(cluster_membership))
	MaskOfClusters(fill(true, n), fill(true, n), fill(0, n), initstats(n), cluster_membership)
end

function invalidate!(mask::MaskOfClusters, i)
	mask.participate[i] .= false
end

function StatsBase.sample!(m::MaskOfClusters)
	ci = m.cluster_membership
	_mask = sample([true, false], maximum(ci))
	for (i,k) in enumerate(ci)
		m.mask[i] = _mask[k]
	end 
end


function Duff.update!(d::MaskOfClusters, v::AbstractArray)
	s = d.stats
	for i in 1:length(d.mask)
		!d.participate[i] && continue
		f = v[d.outputid[i]]
		j = d.cluster_membership[i]
		Duff.update!(s, f, d.mask[i], j)
	end
end

function normalize_clusterids(x)
	n = length(unique(x))
	isempty(setdiff(1:n, unique(x))) && return(x)
	u = unique(x)
	k2id = Dict([u[i] => i for i in 1:length(u)])
	map(k -> k2id[k], x)
end
