struct MatrixMask <: AbstractListMask
	mask::Mask
end

Mask(ds::ArrayNode{T,M}) where {T<:Matrix, M} =  MatrixMask(Mask(size(ds.data,1)))
Mask(ds::ArrayNode{T,M}, m::ArrayModel, cluster_algorithm, verbose = false) where {T<:Matrix, M} =  Mask(ds)

function prune(ds::ArrayNode{T,M}, m::MatrixMask) where {T<:Matrix, M}
	x = deepcopy(ds.data)
	x[.!mask(m), :] .= 0
	ArrayNode(x, ds.metadata)
end

function invalidate!(mask::MatrixMask, observations::Vector{Int})
end
