struct BagMask{C,B,M} <: AbstractExplainMask
	child::C
	bags::B
	mask::M
end

Flux.@functor(BagMask)

function create_mask_structure(ds::BagNode, model::BagModel, create_mask, cluster)
	(isnothing(ds.data) || nobs(ds.data) == 0) && return(EmptyMask())
	child_mask = create_mask_structure(ds.data, model.im, create_mask, cluster)
	cluster_assignments = cluster(model, ds)
	BagMask(child_mask, ds.bags, create_mask(cluster_assignments))
end

function create_mask_structure(ds::BagNode, create_mask)
	(isnothing(ds.data) || nobs(ds.data) == 0) && return(EmptyMask())
	child_mask = create_mask_structure(ds.data, create_mask)
	BagMask(child_mask, ds.bags, create_mask(nobs(ds.data)))
end

function Base.getindex(m::BagMask, i::Mill.VecOrRange)
    nb, ii = Mill.remapbag(m.bags, i)
    isempty(ii) && return(EmptyMask())
    BagNode(m.child[i], nb, m.mask[ii])
end

function mapmask(f, m::BagMask)
	mapmask(f, m.child)
	f(m.mask)
end


function invalidate!(mask::BagMask, observations::Vector{Int})
	invalid_instances = isempty(observations) ? observations : reduce(vcat, [collect(mask.bags[i]) for i in observations])
	participate(mask)[invalid_instances] .= false
	invalid_instances = unique(vcat(invalid_instances, findall(.!(prunemask(mask) .& participate(mask)))))
	invalidate!(mask.child, invalid_instances)
end

function Base.getindex(ds::BagNode, mask::BagMask, presentobs=fill(true, nobs(ds)))
	if !any(presentobs)
		return(ds[0:-1])
	end
	present_childs = prunemask(mask)[:]
	for (i,b) in enumerate(ds.bags) 
	    presentobs[i] && continue
	    present_childs[b] .= false
	end
	x = ds.data[mask.child, present_childs]
	bags = Mill.adjustbags(ds.bags, present_childs)
	if ismissing(x.data)
		bags.bags .= [0:-1]
	end
	BagNode(x, bags[presentobs])
end

function (m::Mill.BagModel)(x::BagNode, mask::BagMask)
	ismissing(x.data) && return(m.bm(ArrayNode(m.a(x.data, x.bags))))
	xx = ArrayNode(transpose(diffmask(mask)) .* m.im(x.data, mask.child).data)
    m.bm(m.a(xx, x.bags))
end

function (m::Mill.BagModel)(x::BagNode, mask::EmptyMask)
	m(x)
end

_nocluster(m::BagModel, ds::BagNode) = nobs(ds.data)
