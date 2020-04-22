struct ProductMask{C} <: AbstractExplainMask
	childs::C
end

Base.getindex(m::ProductMask, i) = m.childs[i]

mask(::ProductMask) = nothing
participate(::ProductMask) = nothing

function Mask(ds::ProductNode)
	ks = keys(ds.data)
	s = (;[k => Mask(ds.data[k]) for k in ks]...)
	ProductMask(s)
end

function Mask(ds::ProductNode, m::ProductModel, cluster_algorithm, verbose::Bool = false)
	ks = keys(ds.data)
	s = (;[k => Mask(ds.data[k], m.ms[k], cluster_algorithm, verbose) for k in ks]...)
	ProductMask(s)
end

NodeType(::Type{T}) where T <: ProductMask = InnerNode()
children(n::ProductMask) = (; n.childs...)
childrenfields(::Type{ProductMask}) = (:childs,)

function mapmask(f, mask::ProductMask)
	ks = keys(mask.childs)
	s = (;[k => mapmask(f, mask.childs[k]) for k in ks]...)
	(;s...)
end

function invalidate!(mask::ProductMask, observations::Vector{Int})
	for c in mask.childs
		invalidate!(c, observations)
	end
end

function prune(ds::ProductNode, mask::ProductMask)
	ks = keys(ds.data)
	s = (;[k => prune(ds.data[k], mask.childs[k]) for k in ks]...)
	ProductNode(s)
end
