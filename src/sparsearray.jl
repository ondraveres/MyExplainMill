struct SparseArrayMask <: AbstractExplainMask
	mask::Array{Bool,1}
	participate::Array{Bool,1}
	columns::Vector{Int}
end

function Mask(ds::ArrayNode{T,M}) where {T<:SparseMatrixCSC, M}
	columns = findall(!iszero, ds.data);
	columns = [c.I[2] for c in columns]
	SparseArrayMask(fill(true, length(columns)), fill(true, length(columns)), columns)
end

function invalidate!(mask::SparseArrayMask, observations::Vector{Int})
	for (i,c) in enumerate(mask.columns)
		if c ∈ observations
			mask.participate[i] = false
		end
	end
end

# function prune(ds::ArrayNode{T,M}, mask::ArrayMask) where {T<:SparseMatrixCSC, M}
# 	x = deepcopy(ds.data)
# 	x.nzval[.!mask.mask] .= 0
# 	ArrayNode(x, ds.metadata)
# end
