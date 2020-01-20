struct BagMask{C,B} <: AbstractExplainMask
	child::C
	bags::B
	mask::Mask
end

BagMask(child, bags, m::Vector{Bool}) = BagMask(child, bags, Mask(m, fill(true, length(m))))
Mask(ds::BagNode) = BagMask(Mask(ds.data), ds.bags, Mask(nobs(ds.data)))

participate(m::BagMask) = participate(m.mask)
mask(m::BagMask) = mask(m.mask)

function mapmask(f, m::BagMask)
	f(m.child)
	f(m.mask)
end

function invalidate!(mask::BagMask, observations::Vector{Int})
	invalid_instances = isempty(observations) ? observations : reduce(vcat, [collect(mask.bags[i]) for i in observations])
	@show invalid_instances
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
