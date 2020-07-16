struct CategoricalMask{M} <: AbstractListMask
	mask::M
end

Flux.@functor(CategoricalMask)

function Mask(ds::ArrayNode{T,M}, m::ArrayModel, initstats, cluster; verbose::Bool = false) where {T<:Flux.OneHotMatrix, M}
	nobs(ds) == 0 && return(EmptyMask())
	cluster_assignments = cluster(m, ds)
	CategoricalMask(Mask(cluster_assignments, initstats))
end

function Mask(ds::ArrayNode{T,M}, initstats; verbose::Bool = false) where {T<:Flux.OneHotMatrix, M}
	nobs(ds) == 0 && return(EmptyMask())
	CategoricalMask(Mask(nobs(ds.data), initstats))
end

function prune(ds::ArrayNode{T,M}, m::CategoricalMask) where {T<:Flux.OneHotMatrix, M}
	msk = prunemask(m)
	ii = map(enumerate(ds.data.data)) do (j,i)
		msk[j] ? i.ix : i.of
	end
	x = Flux.onehotbatch(ii, 1:ds.data.height)
	ArrayNode(x, ds.metadata)
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
    ArrayNode(m.m(ds.data) .* transpose(mulmask(mask)))
end

_nocluster(m::ArrayModel, ds::ArrayNode{T,M})  where {T<:Flux.OneHotMatrix, M} = nobs(ds.data)