struct NGramMatrixMask <: AbstractListMask
	mask::Mask
end

NGramMatrixMask(m::Vector{Bool}) = NGramMatrixMask(Mask(m, fill(true, length(m))))

function Mask(ds::ArrayNode{T,M}) where {T<:Mill.NGramMatrix{String}, M}
	NGramMatrixMask(Mask(length(ds.data.s)))
end

function invalidate!(mask::NGramMatrixMask, observations::Vector{Int})
	participate(mask)[observations] .= false
end

function prune(ds::ArrayNode{T,M}, m::NGramMatrixMask) where {T<:Mill.NGramMatrix{String}, M}
	x = deepcopy(ds.data)
	x.s[.!mask(m)] .= ""
	ArrayNode(x, ds.metadata)
end

dsprint(io::IO, n::NGramMatrixMask; pad=[]) = paddedprint(io, "NGramMatrix")