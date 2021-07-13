struct ProductMask{C} <: AbstractNoMask
	childs::C
end

Flux.@functor(ProductMask)

Base.getindex(m::ProductMask, i::Symbol) = m.childs[i]
Base.getindex(m::ProductMask, i::Int) = m.childs[i]

mask(::ProductMask) = nothing
participate(::ProductMask) = nothing


function create_mask_structure(ds::ProductNode{T,M}, m::ProductModel, create_mask, cluster) where {T<:NamedTuple,M}
	ks = keys(ds.data)
	s = (;[k => create_mask_structure(ds.data[k], m.ms[k], create_mask, cluster) for k in ks]...)
	ProductMask(s)
end

function create_mask_structure(ds::ProductNode{T,M}, create_mask) where {T<:NamedTuple,M}
	ks = keys(ds.data)
	s = (;[k => create_mask_structure(ds.data[k], create_mask) for k in ks]...)
	ProductMask(s)
end

function foreach_mask(f, mk::ProductMask, level = 1)
	foreach(m -> foreach_mask(f, m, level), mk.childs)
end

function mapmask(f, mk::ProductMask, level = 1)
	ProductMask(map(m -> mapmask(f, m, level), mk.childs))
end


function invalidate!(mk::ProductMask, observations::Vector{Int})
	for c in mk.childs
		invalidate!(c, observations)
	end
end

function Base.getindex(ds::ProductNode{T,M}, mk::ProductMask, presentobs=fill(true, nobs(ds))) where {T<:NamedTuple, M}
	ks = keys(ds.data)
	s = (;[k => ds.data[k][mk.childs[k], presentobs] for k in ks]...)
	ProductNode(s)
end

function Base.getindex(ds::ProductNode{T,M}, mk::ProductMask, presentobs=fill(true, nobs(ds))) where {T<:Tuple, M}
	s = tuple([ds.data[k][mk.childs[k], presentobs] for k in 1:length(ds.data)]...)
	ProductNode(s)
end

function (m::Mill.ProductModel{MS,M})(x::ProductNode{P,T}, mk::ProductMask) where {P<:NamedTuple,T,MS<:NamedTuple, M} 
    xx = ArrayNode(vcat([m[k](x[k], mk[k]).data for k in keys(m.ms)]...))
    m.m(xx)
end

function (m::Mill.ProductModel{MS,M})(x::ProductNode{P,T}, mk::ProductMask) where {P<:Tuple,T,MS<:Tuple, M} 
    xx = vcat([m.ms[k](x.data[k], mk.childs[k]) for k in 1:length(m.ms)]...)
    m.m(xx)
end

_nocluster(m::ProductModel, ds::ProductNode) = nobs(ds)

prunemask(m::ProductMask) = nothing
