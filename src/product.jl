struct TreeMask{C}
	childs::C
end

function Mask(ds::TreeNode)
	ks = keys(ds.data)
	s = (;[k => Mask(ds.data[k]) for k in ks]...)
	TreeMask(s)
end

function invalidate!(mask::TreeMask, observations::Vector{Int})
	for c in mask.childs
		invalidate!(c, observations)
	end
end

function prune(ds::TreeNode, mask::TreeMask)
	ks = keys(ds.data)
	s = (;[k => prune(ds.data[k], mask.child_masks[k]) for k in ks]...)
	TreeNode(s)
end
