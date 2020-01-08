struct TreeMask{C}
	child_masks::C
end

struct TreeDaf{C}
	childs::C
end

mask_length(ds::TreeNode) = 0

function StatsBase.sample(daf::TreeDaf, ds::TreeNode)
	ks = keys(ds.data)
	s = (;[k => sample(daf.childs[k], ds.data[k]) for k in ks]...)
	return(TreeNode((;[k => s[k][1] for k in ks]...)), TreeMask((;[k => s[k][2] for k in ks]...)))
end

function Duff.Daf(ds::TreeNode)
	ks = keys(ds.data)
	s = (;[k => Duff.Daf(ds.data[k]) for k in ks]...)
	TreeDaf(s)
end

function Duff.update!(daf::TreeDaf, mask::TreeMask, v::Number, valid_indexes = nothing)
	for k in keys(mask.child_masks)
		Duff.update!(daf.childs[k], mask.child_masks[k], v, valid_indexes)
	end
end
