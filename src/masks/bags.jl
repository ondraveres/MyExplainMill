struct BagMask{C,B,M} <: AbstractExplainMask
	child::C
	bags::B
	mask::M
end

Flux.@functor(BagMask)

function Mask(ds::BagNode, m::BagModel, initstats, cluster; verbose::Bool = false)
	isnothing(ds.data) && return(EmptyMask())
	nobs(ds.data) == 0 && return(EmptyMask())
	child_mask = Mask(ds.data, m.im, initstats, cluster, verbose = verbose)
	cluster_assignments = cluster(m.im, ds.data)
	if verbose
		n, m = nobs(ds.data), length(unique(cluster_assignments))
		println("number of instances: ", n, " ratio: ", round(m/n, digits = 3))
	end
	BagMask(child_mask, ds.bags, Mask(cluster_assignments, initstats))
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

function (m::Mill.BagModel)(x::BagNode, mask::ExplainMill.BagMask)
	ismissing(x.data) && return(m.bm(ArrayNode(m.a(x.data, x.bags))))
	xx = ArrayNode(transpose(gnnmask(mask)) .* m.im(x.data, mask.child).data)
    m.bm(m.a(xx, x.bags))
end

index_in_parent(m::ExplainMill.BagMask, i) = only(findall(map(b -> i ∈ b, m.bags)))

_nocluster(m::BagModel, ds::BagNode) = nobs(ds.data)