struct ProductMask{C} <: AbstractNoMask
	childs::C
end

Flux.@functor(ProductMask)

Base.getindex(m::ProductMask, i::Symbol) = m.childs[i]
Base.getindex(m::ProductMask, i::Int) = m.childs[i]

mask(::ProductMask) = nothing
participate(::ProductMask) = nothing


function Mask(ds::ProductNode{T,M}, m::ProductModel, initstats, cluster; verbose::Bool = false) where {T<:NamedTuple,M}
	ks = keys(ds.data)
	s = (;[k => Mask(ds.data[k], m.ms[k], initstats, cluster, verbose = verbose) for k in ks]...)
	ProductMask(s)
end

function Mask(ds::ProductNode{T,M}, m::ProductModel, initstats, cluster; verbose::Bool = false) where {T<:Tuple,M}
	s = tuple([Mask(ds.data[k], m.ms[k], initstats, cluster, verbose = verbose) for k in 1:length(ds.data)]...)
	ProductMask(s)
end

function Mask(ds::ProductNode{T,M}, initstats; verbose::Bool = false) where {T<:NamedTuple,M}
	ks = keys(ds.data)
	s = (;[k => Mask(ds.data[k], initstats, verbose = verbose) for k in ks]...)
	ProductMask(s)
end


function Mask(ds::ProductNode{T,M}, initstats; verbose::Bool = false) where {T<:Tuple,M}
	s = tuple([Mask(ds.data[k], initstats, verbose = verbose) for k in 1:length(ds.data)]...)
	ProductMask(s)
end

NodeType(::Type{T}) where T <: ProductMask = InnerNode()
children(n::ProductMask) = n.childs
childrenfields(::Type{ProductMask}) = (:childs,)

function mapmask(f, mask::ProductMask{T}) where {T<:NamedTuple}
	ks = keys(mask.childs)
	s = (;[k => mapmask(f, mask.childs[k]) for k in ks]...)
	(;s...)
end

function mapmask(f, mask::ProductMask{T}) where {T<:Tuple}
	map(i -> mapmask(f, i), mask.childs)
end

function invalidate!(mask::ProductMask, observations::Vector{Int})
	for c in mask.childs
		invalidate!(c, observations)
	end
end

function prune(ds::ProductNode{T,M}, mask::ProductMask) where {T<:NamedTuple, M}
	ks = keys(ds.data)
	s = (;[k => prune(ds.data[k], mask.childs[k]) for k in ks]...)
	ProductNode(s)
end

function prune(ds::ProductNode{T,M}, mask::ProductMask) where {T<:Tuple, M}
	s = tuple([prune(ds.data[k], mask.childs[k]) for k in 1:length(ds.data)]...)
	ProductNode(s)
end

function Base.getindex(ds::ProductNode{T,M}, mask::ProductMask, presentobs=fill(true, nobs(ds))) where {T<:NamedTuple, M}
	ks = keys(ds.data)
	s = (;[k => ds.data[k][mask.childs[k], presentobs] for k in ks]...)
	ProductNode(s)
end

prunemask(m::ProductMask) = nothing

function Base.getindex(ds::ProductNode{T,M}, mask::ProductMask, presentobs=fill(true, nobs(ds))) where {T<:Tuple, M}
	s = tuple([ds.data[k][mask.childs[k], presentobs] for k in 1:length(ds.data)]...)
	ProductNode(s)
end

function (m::Mill.ProductModel{MS,M})(x::ProductNode{P,T}, mask::ExplainMill.ProductMask) where {P<:NamedTuple,T,MS<:NamedTuple, M} 
    xx = vcat([m[k](x[k], mask[k]) for k in keys(m.ms)]...)
    m.m(xx)
end

function (m::Mill.ProductModel{MS,M})(x::ProductNode{P,T}, mask::ExplainMill.ProductMask) where {P<:Tuple,T,MS<:Tuple, M} 
    xx = vcat([m.ms[k](x.data[k], mask.childs[k]) for k in 1:length(m.ms)]...)
    m.m(xx)
end

_nocluster(m::ProductModel, ds::ProductNode) = nobs(ds)