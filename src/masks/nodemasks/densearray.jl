"""
	struct MatrixMask{M} <: AbstractListMask
		mask::M
		rows::Int
		cols::Int
	end


	Explaining individual items of Dense matrices. Each item within the matrix has its own mask.
	It is assumed and 

"""
struct MatrixMask{M} <: AbstractListMask
	mask::M
	rows::Int
	cols::Int
end

Flux.@functor(MatrixMask)

function create_mask_structure(ds::ArrayNode{T,M}, m::ArrayModel, create_mask, cluster) where {T<:Matrix, M} 
	create_mask_structure(ds, create_mask)
end

function create_mask_structure(ds::ArrayNode{T,M}, create_mask) where {T<:Matrix, M} 
	MatrixMask(create_mask(size(ds.data, 1)), size(ds.data)...)
end

function Base.getindex(ds::ArrayNode{T,M}, mk::MatrixMask, presentobs=fill(true,nobs(ds))) where {T<:Matrix, M}
	x = ds.data[:,presentobs]
	x[.!prunemask(mk.mask), :] .= 0
	ArrayNode(x, ds.metadata)
end

function Base.getindex(ds::ArrayNode{T,M}, mk::ObservationMask, presentobs=fill(true,nobs(ds))) where {T<:Matrix, M}
	x = ds.data[:, presentobs]
	x[:, (.!prunemask(mk.mask))[presentobs]] .= 0
	ArrayNode(x, ds.metadata)
end

function foreach_mask(f, m::MatrixMask, level, visited)
	if !haskey(visited, m.mask)
		f(m.mask, level)
		visited[m.mask] = nothing
	end
end

function mapmask(f, m::MatrixMask, level, visited)
	new_mask = get!(visited, m.mask, f(m.mask, level))
	MatrixMask(new_mask, m.rows, m.cols)
end

function invalidate!(mk::MatrixMask, observations::Vector{Int})
	
end

function present(mk::MatrixMask, obs)
	any(prunemask(mk.mask)) && return(obs)
	fill(false, length(obs))
end

function (m::Mill.ArrayModel)(ds::ArrayNode{<:Matrix,<:Any}, mk::MatrixMask)
    ArrayNode(m.m(diffmask(mk.mask) .* ds.data))
end

function (m::Mill.ArrayModel)(ds::ArrayNode{<:Matrix,<:Any}, mk::ObservationMask)
    ArrayNode(m.m(transpose(diffmask(mk.mask)) .* ds.data))
end

"""
	Mill.partialeval(model::AbstracModel, ds::AbstractNode, mk::StructureMask, masks)

	identify subset of `model`, sample `ds`, and structural mask `mk` that are 
	sensitive to `masks` and evaluate and replace the rest of the model, sample, 
	and masks by `identity`, `ArrayNode`, and `EmptyMask`. Partial evaluation 
	is useful when we are explaining only subset of full mask (e.g. level-by-level) 
	explanation.
"""
function Mill.partialeval(model::M, ds::ArrayNode, mk, masks) where {M<:Union{IdentityModel, ArrayModel}}
	mk ∈ masks && return(model, ds, mk, true)
	mk.mask ∈ masks && return(model, ds, mk, true)
	return(ArrayModel(identity), model(ds), EmptyMask(), false)
end

_nocluster(m::ArrayModel, ds::ArrayNode{T,M}) where {T<:Matrix,M} = size(ds.data, 2)