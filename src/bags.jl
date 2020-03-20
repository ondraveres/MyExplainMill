struct BagMask{C,B} <: AbstractExplainMask
	child::C
	bags::B
	mask::Array{Bool,1}
	participate::Array{Bool,1}
end

Mask(ds::BagNode) = BagMask(Mask(ds.data), ds.bags, fill(true, nobs(ds.data)), fill(true, nobs(ds.data)))

function invalidate!(mask::BagMask, observations::Vector{Int})
	invalid_instances = isempty(observations) ? observations : reduce(vcat, [collect(mask.bags[i]) for i in observations])
	@show invalid_instances
	mask.participate[invalid_instances] .= false
	invalid_instances = unique(vcat(invalid_instances, findall(.!mask.mask)))
	invalidate!(mask.child, invalid_instances)
end

function prune(ds::BagNode, mask::BagMask)
	x = prune(ds.data, mask.child_masks)
	x = Mill.subset(x, findall(mask.mask))
	bags = Mill.adjustbags(ds.bags, mask.mask)
	BagNode(x, bags)
end
