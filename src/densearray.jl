
struct Explainer end


struct ArrayMask{M}
	mask::M
end

struct ArrayDaf{M} <: AbstractDaf
	daf::M
end

mask_length(ds::ArrayNode{T, A}) where {T<:Matrix, A} = size(ds.data,1)

function StatsBase.sample(daf::ArrayDaf, ds::ArrayNode{T, A}) where {T<:Matrix, A}
	x = deepcopy(ds.data)
	mask = rand(mask_length(ds)) .>= 0.5
	x[.!mask, :] .= 0
	return(ArrayNode(x), ArrayMask(mask))
end

StatsBase.sample(e::Explainer, ds::ArrayNode) = return(ds, ArrayMask(BitArray{1}()))

function Duff.Daf(ds::ArrayNode{T,M}) where {T<:Matrix, M}
	ArrayDaf(Duff.Daf(mask_length(ds)))
end

function Duff.update!(daf::ArrayDaf, mask::ArrayMask, v::Number, valid_indexes = nothing)
	Duff.update!(daf.daf, mask.mask, v)
end


dsprint(io::IO, n::ArrayDaf; pad=[]) = paddedprint(io, "DenseArray")