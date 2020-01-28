struct NGramMatrixMask <: AbstractListMask
	mask::Mask
end

function Mask(ds::ArrayNode{T,M}) where {T<:Mill.NGramMatrix{String}, M}
	NGramMatrixMask(Mask(length(ds.data.s)))
end

function Mask(ds::ArrayNode{T,M}, m::ArrayModel; cluster_algorithm = cluster_instances, verbose = false) where {T<:Mill.NGramMatrix{String}, M}
	cluster_assignments = cluster_algorithm(m(ds).data)
	if verbose
		n, m = nobs(ds), length(unique(cluster_assignments)), length(unique(ds.data.s))
		println("number of strings: ", n, " number of clusters: ", m, " ratio: ", round(m/n, digits = 3))
	end
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