struct Mask{I<:Union{Nothing, Vector{Int}}}
	mask::Array{Bool,1}
	participate::Array{Bool,1}
	daf::Daf
	cluster_membership::I
end

participate(m::Mask) = m.participate
mask(m::Mask) = m.mask

Base.getindex(m::Mask{Nothing}, i::Int) = m.mask[i]
Base.getindex(m::Mask{Vector{Int}}, i::Int) = m.mask[m.cluster_membership .== i]
Base.setindex!(m::Mask{Nothing}, v, i::Int) = m.mask[i] = v
Base.setindex!(m::Mask{Vector{Int}}, v, i::Int) = m.mask[m.cluster_membership .== i] .= v

####
#	Explaination without clustering, where each item is independent of others
####
Mask(d::Int) = Mask(fill(true, d), fill(true, d), Daf(d), nothing)

function Duff.update!(d::Mask{M}, v) where {M<:Nothing}
	Duff.update!(d.daf, v, d.mask, d.participate)
end

function StatsBase.sample!(m::Mask{Nothing})
	m.mask .= sample([true, false], length(m.mask))
end


####
#	Explaination, where items are clustered together
####
function Mask(cluster_membership::Vector{Int}) 
	n = length(unique(cluster_membership))
	!isempty(setdiff(1:n, unique(cluster_membership))) && @error "clusters should be labeled from 1 to n"
	d = length(cluster_membership)
	Mask(fill(true, d), fill(true, d), Daf(n), cluster_membership)
end

function Duff.update!(d::Mask{M}, v) where {M<:Vector{Int64}}
	Duff.update!(d.daf, v, d.mask, d.participate, d.cluster_membership)
end

function StatsBase.sample!(m::Mask{Vector{Int64}})
	ci = m.cluster_membership
	_mask = sample([true, false], maximum(ci))
	for (i,k) in enumerate(ci)
		m.mask[i] = _mask[k]
	end 
end
