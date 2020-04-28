struct MatrixMask{M} <: AbstractListMask
	mask::M
end

Flux.@functor(MatrixMask)

function Mask(ds::ArrayNode{T,M}, m::ArrayModel, initstats, cluster; verbose::Bool = false) where {T<:Matrix, M} 
	Mask(nobs(ds.data), initstats)
end

function prune(ds::ArrayNode{T,M}, m::MatrixMask) where {T<:Matrix, M}
	x = deepcopy(ds.data)
	x[.!prunemask(m)[:], :] .= 0
	ArrayNode(x, ds.metadata)
end

function invalidate!(mask::MatrixMask, observations::Vector{Int})
end

function (m::Mill.ArrayModel)(ds::ArrayNode, mask::MatrixMask)
    ArrayNode(m.m(mask.mask .* ds.data))
end

index_in_parent(m::MatrixMask, i) = 1