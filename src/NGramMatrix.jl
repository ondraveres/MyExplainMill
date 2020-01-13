struct NGramMatrixDaf{M} <: AbstractDaf
	daf::M
end

mask_length(ds::ArrayNode{T, A}) where {T<:Mill.NGramMatrix{String}, A} = length(ds.data.s)

function StatsBase.sample(daf::NGramMatrixDaf, ds::ArrayNode{T, A}) where {T<:Mill.NGramMatrix{String}, A}
	mask_length(ds) == 0 && return(ds, ArrayMask(BitArray{1}()))
	x = deepcopy(ds.data)
	mask = rand(mask_length(ds)) .>= 0.5
	x.s[.!mask] .= ""
	return(ArrayNode(x), ArrayMask(mask))
end

function Duff.Daf(ds::ArrayNode{T,M}) where {T<:Mill.NGramMatrix{String}, M}
	NGramMatrixDaf(Duff.Daf(mask_length(ds)))
end

function Duff.update!(daf::NGramMatrixDaf, mask::ArrayMask, v::Number, valid_columns)
	Duff.update!(daf.daf, mask.mask, v, valid_columns)
end

function Duff.update!(daf::NGramMatrixDaf, mask::ArrayMask, v::Number, valid_columns::Nothing) 
	Duff.update!(daf, mask, v)
end

function Duff.update!(daf::NGramMatrixDaf, mask::ArrayMask, v::Number)
	Duff.update!(daf.daf, mask.mask, v)
end

function prune(ds::ArrayNode{T,M}, mask::ArrayMask) where {T<:Mill.NGramMatrix{String}, M}
	x = deepcopy(ds.data)
	x.s[.!mask.mask] .= ""
	ArrayNode(x, ds.metadata)
end

function masks_and_stats(daf::NGramMatrixDaf, depth = 0)
	mask = fill(true, length(daf.daf))
	return(ArrayMask(mask), [(m = mask, d = daf.daf, depth = depth)])
end

dsprint(io::IO, n::NGramMatrixDaf; pad=[]) = paddedprint(io, "NGramMatrix")