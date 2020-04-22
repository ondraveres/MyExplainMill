struct CategoricalMask <: AbstractListMask
	mask::Mask
end

Mask(ds::ArrayNode{T,M}) where {T<:Flux.OneHotMatrix, M} =  CategoricalMask(Mask(size(ds.data,2)))

function Mask(ds::ArrayNode{T,M}, m::ArrayModel, cluster_algorithm, verbose::Bool = false) where {T<:Flux.OneHotMatrix, M}
	nobs(ds) == 0 && return(EmptyMask())
	# cluster_assignments = cluster_algorithm(m(ds).data)
	cluster_assignments = cluster_algorithm(m, ds)
	CategoricalMask(Mask(cluster_assignments))
end

function prune(ds::ArrayNode{T,M}, m::CategoricalMask) where {T<:Flux.OneHotMatrix, M}
	ii = map(enumerate(ds.data.data)) do ji
		j,i = ji
		mask(m)[j] ? i.ix : i.of
	end
	x = Flux.onehotbatch(ii, 1:ds.data.height)
	ArrayNode(x, ds.metadata)
end

function invalidate!(mask::CategoricalMask, observations::Vector{Int})
	participate(mask.mask)[observations] .= false
end
