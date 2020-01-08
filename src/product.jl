struct TreeMask{C}
	child_masks::C
end

struct TreeDaf{C} <: AbstractDaf
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


function dsprint(io::IO, n::TreeDaf; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "Tree", color=c)
    m = length(n.childs)
    ks = keys(n.childs)
    for i in 1:m
    	k = "$(ks[i]): "
        println(io)
        if i < m
	        paddedprint(io, "  ├── $(k)", color=c, pad=pad)
	        dsprint(io, n.childs[i], pad=[pad; (c, "  │" * repeat(" ", max(3, 2+length(k))))])
	    else
		    paddedprint(io, "  └── $(k)", color=c, pad=pad)
		    dsprint(io, n.childs[end], pad=[pad; (c, repeat(" ", 3+max(3, 2+length(k))))])
	    end
    end
end


