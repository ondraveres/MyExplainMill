struct TreeMask{C}
	childs::C
end

mask(::TreeMask) = nothing
participate(::TreeMask) = nothing

function Mask(ds::TreeNode)
	ks = keys(ds.data)
	s = (;[k => Mask(ds.data[k]) for k in ks]...)
	TreeMask(s)
end

function mapmask(f, mask::TreeMask)
	ks = keys(mask.childs)
	s = (;[k => f(mask.childs[k]) for k in ks]...)
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

function dsprint(io::IO, n::TreeMask; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "TreeMask", color=c)
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


