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

function Base.getindex(ds::BagNode, mk::BagMask, presentobs=fill(true, nobs(ds)))
	if !any(presentobs)
		return(ds[0:-1])
	end
	present_childs = prunemask(mk.mask)[:]
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
    model.bm(model.a(xx, x.bags, mk.mask))
end

#TODO: SimpleMask for now, but we should add a proper abstract
function (a::Mill.SegmentedMax)(x::Matrix, bags::Mill.AbstractBags, mk::AbstractVectorMask)
	m = transpose(diffmask(mk))
	xx = m .* x .+ (1 .- m) .* a.C 
	a(xx, bags)
end	

# TODO: This might be done better
function (a::Mill.SegmentedMean)(x::Matrix, bags::Mill.AbstractBags, mk::AbstractVectorMask)
	m = transpose(diffmask(mk))
	xx = m .* x
	o = a(xx, bags)
	n = max.(a(eltype(x).(m), bags), 1f-6)
	o ./ n
end	

function (m::Mill.BagModel)(x::BagNode, mk::EmptyMask)
	m(x)
end

_nocluster(m::BagModel, ds::BagNode) = nobs(ds.data)
