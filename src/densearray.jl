
struct MatrixMask <:AbstractExplainMask
	mask::Array{Bool,1}
	participate::Array{Bool,1}
end


Mask(ds::ArrayNode{T,M}) where {T<:Matrix, M} =  MatrixMask(fill(true, size(ds.data, 1)), fill(true, size(ds.data, 1)))

function prune(ds::ArrayNode{T,M}, mask::MatrixMask) where {T<:Matrix, M}
	x = deepcopy(ds.data)
	x[.!mask.mask, :] .= 0
	ArrayNode(x, ds.metadata)
end

function invalidate!(mask::MatrixMask, observations::Vector{Int})
end
