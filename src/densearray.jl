struct MatrixMask <:AbstractListMask
	mask::Mask
end

MatrixMask(m::Vector{Bool}) = MatrixMask(Mask(m, fill(true, length(m))))

Mask(ds::ArrayNode{T,M}) where {T<:Matrix, M} =  MatrixMask(Mask(size(ds.data,1)))

function prune(ds::ArrayNode{T,M}, m::MatrixMask) where {T<:Matrix, M}
	x = deepcopy(ds.data)
	x[.!mask(m), :] .= 0
	ArrayNode(x, ds.metadata)
end

function invalidate!(mask::MatrixMask, observations::Vector{Int})
end



dsprint(io::IO, n::MatrixMask; pad=[]) = paddedprint(io, "MatrixMask")