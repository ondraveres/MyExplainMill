struct TreeMask{C} <: AbstractExplainMask
	childs::C
end

mask(::TreeMask) = nothing
participate(::TreeMask) = nothing

function Mask(ds::TreeNode)
	ks = keys(ds.data)
	s = (;[k => Mask(ds.data[k]) for k in ks]...)
	TreeMask(s)
end

function Mask(ds::TreeNode, m::ProductModel; verbose = false, cluster_algorithm = cluster_instances)
	ks = keys(ds.data)
	s = (;[k => Mask(ds.data[k], m.ms[k], cluster_algorithm = cluster_instances, verbose = verbose) for k in ks]...)
	TreeMask(s)
end

function mapmask(f, mask::TreeMask)
	ks = keys(mask.childs)
	s = (;[k => mapmask(f, mask.childs[k]) for k in ks]...)
	(;s...)
end

function invalidate!(mask::TreeMask, observations::Vector{Int})
	for c in mask.childs
		invalidate!(c, observations)
	end
end

function prune(ds::TreeNode, mask::TreeMask)
	ks = keys(ds.data)
	s = (;[k => prune(ds.data[k], mask.childs[k]) for k in ks]...)
	TreeNode(s)
end
