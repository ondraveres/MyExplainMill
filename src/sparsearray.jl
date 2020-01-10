struct SparseArrayDaf{M} <: AbstractDaf
	daf::M
	columns::Vector{Int}
end

mask_length(ds::ArrayNode{T, A}) where {T<:SparseMatrixCSC, A} = nnz(ds.data)

function StatsBase.sample(daf::SparseArrayDaf, ds::ArrayNode{T, A}) where {T<:SparseMatrixCSC, A}
	mask_length(ds) == 0 && return(ds, ArrayMask(BitArray{1}()))
	x = deepcopy(ds.data)
	mask = rand(mask_length(ds)) .>= 0.5
	x.nzval[.!mask] .= 0
	return(ArrayNode(x), ArrayMask(mask))
end

function Duff.Daf(ds::ArrayNode{T,M}) where {T<:SparseMatrixCSC, M}
	columns = findall(!iszero, ds.data);
	columns = [c.I[2] for c in columns]
	SparseArrayDaf(Duff.Daf(length(columns)), columns)
end

function Duff.update!(daf::SparseArrayDaf, mask::ArrayMask, v::Number, valid_columns)
	valid_columns = findall([i ∈ valid_columns for i in daf.columns])
	Duff.update!(daf.daf, mask.mask, v, valid_columns)
end

function Duff.update!(daf::SparseArrayDaf, mask::ArrayMask, v::Number, valid_columns::Nothing) 
	Duff.update!(daf, mask, v)
end

function Duff.update!(daf::SparseArrayDaf, mask::ArrayMask, v::Number)
	Duff.update!(daf.daf, mask.mask, v)
end

function prune(ds::ArrayNode{T,M}, mask::ArrayMask) where {T<:SparseMatrixCSC, M}
	x = deepcopy(ds.data)
	x.nzval[.!mask.mask] .= 0
	ArrayNode(x, ds.metadata)
end

function masks_and_stats(daf::SparseArrayDaf, depth = 0)
	mask = fill(true, length(daf.daf))
	return(ArrayMask(mask), [(m = mask, d = daf.daf, depth = depth)])
end

dsprint(io::IO, n::SparseArrayDaf; pad=[]) = paddedprint(io, "SparseArray")