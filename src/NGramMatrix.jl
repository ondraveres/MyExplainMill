struct NGramMatrixMask <: AbstractExplainMask
	mask::Array{Bool,1}
	participate::Array{Bool,1}
end

function Mask(ds::ArrayNode{T,M}) where {T<:Mill.NGramMatrix{String}, M}
	NGramMatrixMask(fill(true, length(ds.data.s)), fill(true, length(ds.data.s)))
end

function invalidate!(mask::NGramMatrixMask, observations::Vector{Int})
	mask.participate[observations] .= false
end

# function prune(ds::ArrayNode{T,M}, mask::ArrayMask) where {T<:Mill.NGramMatrix{String}, M}
# 	x = deepcopy(ds.data)
# 	x.s[.!mask.mask] .= ""
# 	ArrayNode(x, ds.metadata)
# end

dsprint(io::IO, n::NGramMatrixMask; pad=[]) = paddedprint(io, "NGramMatrix")