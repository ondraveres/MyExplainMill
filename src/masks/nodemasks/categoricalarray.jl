using SparseArrays

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
	invalidate!(mk.mask, observations)
end

function present(mk::CategoricalMask, obs)
	map((&), obs, prunemask(mk.mask))
end

function foreach_mask(f, m::CategoricalMask, level = 1)
	f(m.mask, level)
end

function mapmask(f, m::CategoricalMask, level = 1)
	CategoricalMask(f(m.mask, level))
end

# This might be actually simplified if we define gradient with respect to ds[mk]
function (m::Mill.ArrayModel)(ds::ArrayNode, mk::CategoricalMask)
	x = Zygote.@ignore sparse(ds.data)
	y = Zygote.@ignore sparse(fill(size(x)...), collect(1:size(x,2)), 1)
	dm = reshape(diffmask(mk.mask), 1, :)
    m(ArrayNode(@. dm * x + (1 - dm) * y))
end

_nocluster(m::ArrayModel, ds::ArrayNode{T,M})  where {T<:Flux.OneHotMatrix, M} = nobs(ds.data)