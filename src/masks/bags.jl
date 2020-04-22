struct BagMask{C,B} <: AbstractExplainMask
	child::C
	bags::B
	mask::Mask
end

Mask(ds::BagNode) = BagMask(Mask(ds.data), ds.bags, Mask(nobs(ds.data)))

function Mask(ds::BagNode{Missing, B,M})  where {B<:Mill.AbstractBags,M}
	return(EmptyMask())
end

function Mask(ds::BagNode{Missing, B,M}, m, cluster_algorithm, verbose::Bool = false)  where {B<:Mill.AbstractBags,M}
	return(EmptyMask())
end

function Mask(ds::BagNode{Missing, B,M}, m::BagModel, cluster_algorithm, verbose::Bool = false)  where {B<:Mill.AbstractBags,M}
	return(EmptyMask())
end

function Mask(ds::BagNode, m::BagModel, cluster_algorithm, verbose::Bool = false)
	nobs(ds.data) == 0 && return(EmptyMask())
	child_mask = Mask(ds.data, m.im, cluster_algorithm, verbose)
	# cluster_assignments = cluster_algorithm(m.im(ds.data).data)
	cluster_assignments = cluster_algorithm(m.im, ds.data)
	if verbose
		n, m = nobs(ds.data), length(unique(cluster_assignments))
		println("number of instances: ", n, " ratio: ", round(m/n, digits = 3))
	end
	BagMask(child_mask, ds.bags, Mask(cluster_assignments))
end

NodeType(::Type{T}) where T <: BagMask = SingletonNode()
children(n::BagMask) = (n.child,)
childrenfields(::Type{T}) where T <: BagMask = (:child,)


function mapmask(f, m::BagMask)
	mapmask(f, m.child)
	f(m.mask)
end

function invalidate!(mask::BagMask, observations::Vector{Int})
	invalid_instances = isempty(observations) ? observations : reduce(vcat, [collect(mask.bags[i]) for i in observations])
	mask.mask.participate[invalid_instances] .= false
	invalid_instances = unique(vcat(invalid_instances, findall(.!mask.mask.mask)))
	invalidate!(mask.child, invalid_instances)
end

function prune(ds::BagNode, mask::BagMask)
	x = prune(ds.data, mask.child)
	x = Mill.subset(x, findall(mask.mask.mask))
	bags = Mill.adjustbags(ds.bags, mask.mask.mask)
	if ismissing(x.data)
		bags.bags .= [0:-1]
	end
	BagNode(x, bags)
end
