struct SparseArrayMask <: AbstractListMask
	mask::Mask
	columns::Vector{Int}
end

function Mask(ds::ArrayNode{T,M}) where {T<:SparseMatrixCSC, M}
	columns = findall(!iszero, ds.data);
	columns = [c.I[2] for c in columns]
	SparseArrayMask(Mask(length(columns)), columns)
end

function Mask(ds::ArrayNode{T,M}, m::ArrayModel, cluster_algorithm = cluster_instances) where {T<:SparseMatrixCSC, M}
	Mask(ds)
end

function invalidate!(mask::SparseArrayMask, observations::Vector{Int})
	for (i,c) in enumerate(mask.columns)
		if c ∈ observations 
			mask.mask.participate[i] = false
		end
	end
end

function prune(ds::ArrayNode{T,M}, mask::SparseArrayMask) where {T<:SparseMatrixCSC, M}
	x = deepcopy(ds.data)
	x.nzval[.!mask.mask.mask] .= 0
	ArrayNode(x, ds.metadata)
end

dsprint(io::IO, n::SparseArrayMask; pad=[]) = paddedprint(io, "SparseArrayMask")