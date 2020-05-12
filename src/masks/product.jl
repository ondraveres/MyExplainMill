struct ProductMask{C} <: AbstractNoMask
	childs::C
end

Flux.@functor(ProductMask)

Base.getindex(m::ProductMask, i) = m.childs[i]

mask(::ProductMask) = nothing
participate(::ProductMask) = nothing

function Mask(ds::ProductNode, m::ProductModel, initstats, cluster; verbose::Bool = false)
	ks = keys(ds.data)
	s = (;[k => Mask(ds.data[k], m.ms[k], initstats, cluster, verbose = verbose) for k in ks]...)
	ProductMask(s)
end

function Mask(ds::ProductNode, initstats; verbose::Bool = false)
	ks = keys(ds.data)
	s = (;[k => Mask(ds.data[k], initstats, verbose = verbose) for k in ks]...)
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

function (m::Mill.ProductModel{MS,M})(x::ProductNode{P,T}, mask::ExplainMill.ProductMask) where {P<:NamedTuple,T,MS<:NamedTuple, M} 
    xx = vcat([m[k](x[k], mask[k]) for k in keys(m.ms)]...)
    m.m(xx)
end

_nocluster(m::ProductModel, ds::ProductNode) = nobs(ds)