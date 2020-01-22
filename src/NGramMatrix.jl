struct NGramMatrixMask <: AbstractListMask
	mask::Mask
end

function Mask(ds::ArrayNode{T,M}) where {T<:Mill.NGramMatrix{String}, M}
	NGramMatrixMask(Mask(length(ds.data.s)))
end

function Mask(ds::ArrayNode, m::ArrayModel)
	cluster_assignments = m(ds).data
	NGramMatrixMask(Mask(cluster_assignments))
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