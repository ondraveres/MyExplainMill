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

function foreach_mask(f, m::MatrixMask, level = 1)
	f(m.mask, level)
end

function mapmask(f, m::MatrixMask, level = 1)
	MatrixMask(f(m.mask, level), m.rows, m.cols)
end

function invalidate!(mk::MatrixMask, observations::Vector{Int})
	
end

function (m::Mill.ArrayModel)(ds::ArrayNode, mk::MatrixMask)
    ArrayNode(m.m(diffmask(mk.mask) .* ds.data))
end

function Mill.partialeval(model::M, ds::ArrayNode, mk, masks) where {M<:Union{IdentityModel, ArrayModel}}
	mk ∈ masks && return(model, ds, mk, true)
	mk.mask ∈ masks && return(model, ds, mk, true)
	return(ArrayModel(identity), model(ds), EmptyMask(), false)
end

_nocluster(m::ArrayModel, ds::ArrayNode{T,M}) where {T<:Matrix,M} = size(ds.data, 2)