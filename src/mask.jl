struct Mask{I<:Union{Nothing, Vector{Int}}}
	mask::Array{Bool,1}
	participate::Array{Bool,1}
	cluster_membership::I
end

Mask(d::Int) = Mask(fill(true, d), fill(true, d), nothing)

function Mask(cluster_membership::Vector{Int}) 
	d = length(unique(cluster_membership))
	!isempty(setdiff(1:d, unique(cluster_membership))) && @error "clusters should be labeled from 1 to n"
	Mask(fill(true, d), fill(true, d), cluster_membership)
end

participate(m::Mask) = m.participate
mask(m::Mask) = m.mask