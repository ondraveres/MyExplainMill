struct CategoricalMask{M} <: AbstractListMask
	mask::M
end

Flux.@functor(CategoricalMask)

function create_mask_structure(ds::ArrayNode{T,M}, m::ArrayModel, create_mask, cluster) where {T<:Flux.OneHotMatrix, M}
	nobs(ds) == 0 && return(EmptyMask())
	cluster_assignments = cluster(m, ds)
	CategoricalMask(create_mask(cluster_assignments))
end

function create_mask_structure(ds::ArrayNode{T,M}, create_mask) where {T<:Flux.OneHotMatrix, M}
	nobs(ds) == 0 && return(EmptyMask())
	CategoricalMask(create_mask(nobs(ds.data)))
end

function Base.getindex(ds::ArrayNode{T,M}, mk::CategoricalMask, presentobs=fill(true, nobs(ds))) where {T<:Flux.OneHotMatrix, M}
	pm = prunemask(mk.mask)
	ii = map(findall(presentobs)) do j
		i = ds.data.data[j]
		pm[j] ? i.ix : i.of
	end
	x = Flux.onehotbatch(ii, 1:ds.data.height)
	ArrayNode(x, ds.metadata)
end

function invalidate!(mk::CategoricalMask, observations::Vector{Int})
	participate(mk.mask)[observations] .= false
end

function (m::Mill.ArrayModel)(ds::ArrayNode, mk::CategoricalMask)
    ArrayNode(m.m(ds.data) .* transpose(diffmask(mk.mask)))
end

_nocluster(m::ArrayModel, ds::ArrayNode{T,M})  where {T<:Flux.OneHotMatrix, M} = nobs(ds.data)