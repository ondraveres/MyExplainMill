struct BagMask{C,B} <: AbstractExplainMask
	child::C
	bags::B
	mask::Mask
end

Mask(ds::BagNode) = BagMask(Mask(ds.data), ds.bags, Mask(nobs(ds.data)))

function Mask(ds::BagNode, m::BagModel, cluster_algorithm = cluster_instances)
	child_mask = Mask(ds.data, m.im, cluster_algorithm)
	cluster_assignments = cluster_algorithm(m.im(ds.data).data)
	BagMask(child_mask, ds.bags, Mask(cluster_assignments))
end

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
	BagNode(x, bags)
end

function dsprint(io::IO, n::BagMask; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io,"BagMask\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.child, pad = [pad; (c, "      ")])
end
