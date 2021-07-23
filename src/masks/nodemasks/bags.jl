struct BagMask{C,B,M} <: AbstractStructureMask
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

function foreach_mask(f, m::BagMask, level = 1)
	f(m.mask, level)
	foreach_mask(f, m.child, level + 1)
end

function mapmask(f, m::BagMask, level = 1)
	BagMask(mapmask(f, m.child, level + 1),
		m.bags,
		f(m.mask, level)
		)
end

function invalidate!(mk::BagMask, invalid_observations::AbstractVector{Int})
	invalid_instances = isempty(invalid_observations) ? invalid_observations : reduce(vcat, [collect(mk.bags[i]) for i in invalid_observations])
	invalidate!(mk.mask, invalid_instances)
	invalid_instances = unique(vcat(invalid_instances, findall(.!(prunemask(mk.mask) .& participate(mk.mask)))))
	invalidate!(mk.child, invalid_instances)
end

function present(mk::BagMask, obs)
	child_obs = present(mk.child, prunemask(mk.mask))
	map(enumerate(mk.bags)) do (i, b) 
		obs[i] & any(@view child_obs[b])
	end
end

function Base.getindex(ds::BagNode, mk::BagMask, presentobs=fill(true, nobs(ds)))
	if !any(presentobs)
		return(ds[0:-1])
	end
	present_childs = present(mk.child, prunemask(mk.mask))
	for (i,b) in enumerate(ds.bags) 
	    presentobs[i] && continue
	    present_childs[b] .= false
	end
	x = ds.data[mk.child, present_childs]
	bags = Mill.adjustbags(ds.bags, present_childs)
	if ismissing(x.data)
		bags.bags .= [0:-1]
	end
	BagNode(x, bags[presentobs])
end

function (model::Mill.BagModel)(x::BagNode, mk::BagMask)
	ismissing(x.data) && return(model.bm(ArrayNode(model.a(x.data, x.bags))))
	xx = model.im(x.data, mk.child)
    model.bm(model.a(xx, x.bags, mk))
end

#TODO: SimpleMask for now, but we should add a proper abstract
function (a::Mill.SegmentedMax)(x::Matrix, bags::Mill.AbstractBags, mk::BagMask)
	present_childs = Zygote.@ignore present(mk.child, prunemask(mk.mask))
	m = transpose(diffmask(mk.mask) .* present_childs)
	xx = m .* x .+ (1 .- m) .* a.C 
	a(xx, bags)
end	

# TODO: This might be done better
function (a::Mill.SegmentedMean)(x::Matrix{T}, bags::Mill.AbstractBags, mk::BagMask) where {T}
	present_childs = Zygote.@ignore present(mk.child, prunemask(mk.mask))
	m = T.(transpose(diffmask(mk.mask) .* present_childs))
	xx = m .* x
	o = a(xx, bags)
	n = max.(a(m, bags), eps(T))
	o ./ n
end	

function (m::Mill.BagModel)(x::BagNode, mk::EmptyMask)
	m(x)
end

function Mill.partialeval(model::BagModel, ds::BagNode, mk::BagMask, masks)
	im, ids, childms, keep = Mill.partialeval(model.im, ds.data, mk.child, masks)
	if (mk ∈ masks) | (mk.mask ∈ masks) | keep
		return(BagModel(im, model.a,  model.bm), BagNode(ids, ds.bags, ds.metadata), BagMask(childms, mk.bags, mk.mask), true)
	end
	return(ArrayModel(identity), model.bm(model.a(ids, ds.bags)), EmptyMask(), false)
end

function Mill.partialeval(model::BagModel, ds::BagNode, mk::EmptyMask, masks)
	return(ArrayModel(identity), model(ds), EmptyMask(), false)
end


_nocluster(m::BagModel, ds::BagNode) = nobs(ds.data)
