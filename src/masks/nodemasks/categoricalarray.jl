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

function Base.getindex(ds::ArrayNode{T,M}, m::CategoricalMask, presentobs=fill(true, nobs(ds))) where {T<:Flux.OneHotMatrix, M}
	pm = prunemask(m)
	ii = map(findall(presentobs)) do j
		i = ds.data.data[j]
		pm[j] ? i.ix : i.of
	end
	x = Flux.onehotbatch(ii, 1:ds.data.height)
	ArrayNode(x, ds.metadata)
end

function invalidate!(mask::CategoricalMask, observations::Vector{Int})
	participate(mask.mask)[observations] .= false
end

function (m::Mill.ArrayModel)(ds::ArrayNode, mask::CategoricalMask)
    ArrayNode(m.m(ds.data) .* transpose(diffmask(mask)))
end

_nocluster(m::ArrayModel, ds::ArrayNode{T,M})  where {T<:Flux.OneHotMatrix, M} = nobs(ds.data)