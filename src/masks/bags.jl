struct BagMask{C,B,M} <: AbstractExplainMask
	child::C
	bags::B
	mask::M
end

Flux.@functor(BagMask)

function Mask(ds::BagNode, m::BagModel, initstats, cluster; verbose::Bool = false)
	isnothing(ds.data) && return(EmptyMask())
	nobs(ds.data) == 0 && return(EmptyMask())
	child_mask = Mask(ds.data, m.im, initstats, cluster)
	cluster_assignments = cluster(m, ds)
	if verbose
		n, m = nobs(ds.data), length(unique(cluster_assignments))
		println("number of instances: ", n, " ratio: ", round(m/n, digits = 3))
	end
	BagMask(child_mask, ds.bags, Mask(cluster_assignments, initstats))
end

function Mask(ds::BagNode, initstats; verbose::Bool = false)
	isnothing(ds.data) && return(EmptyMask())
	nobs(ds.data) == 0 && return(EmptyMask())
	child_mask = Mask(ds.data, initstats; verbose = verbose)
	BagMask(child_mask, ds.bags, Mask(nobs(ds.data), initstats))
end

NodeType(::Type{T}) where T <: BagMask = SingletonNode()
children(n::BagMask) = (n.child,)
childrenfields(::Type{T}) where T <: BagMask = (:child,)

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

function prune(ds::BagNode, mask::BagMask)
	x = prune(ds.data, mask.child)
	pmask = prunemask(mask)[:]
	x = Mill.subset(x, findall(pmask))
	bags = Mill.adjustbags(ds.bags, pmask)
	if ismissing(x.data)
		bags.bags .= [0:-1]
	end
	BagNode(x, bags)
end

function Base.getindex(ds::BagNode, mask::BagMask, presentobs=fill(true, nobs(ds)))
	#first, let's find the set of valid childs 
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
	BagNode(x, bags)
end

function (m::Mill.BagModel)(x::BagNode, mask::BagMask)
	ismissing(x.data) && return(m.bm(ArrayNode(m.a(x.data, x.bags))))
	xx = ArrayNode(transpose(mulmask(mask)) .* m.im(x.data, mask.child).data)
    m.bm(m.a(xx, x.bags))
end

function (m::Mill.BagModel)(x::BagNode, mask::EmptyMask)
	m(x)
end

index_in_parent(m::BagMask, i) = only(findall(map(b -> i ∈ b, m.bags)))

_nocluster(m::BagModel, ds::BagNode) = nobs(ds.data)