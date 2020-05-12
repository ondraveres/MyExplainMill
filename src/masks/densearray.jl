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

function Mask(ds::ArrayNode{T,M}, m::ArrayModel, initstats, cluster; verbose::Bool = false) where {T<:Matrix, M} 
	Mask(ds, initstats, verbose = verbose)
end

function Mask(ds::ArrayNode{T,M}, initstats; verbose::Bool = false) where {T<:Matrix, M} 
	MatrixMask(Mask(length(ds.data), initstats), size(ds.data)...)
end

function prune(ds::ArrayNode{T,M}, m::MatrixMask) where {T<:Matrix, M}
	x = deepcopy(ds.data)
	x[.!prunemask(m)] .= 0
	ArrayNode(x, ds.metadata)
end

function invalidate!(mask::MatrixMask, observations::Vector{Int})
	for i in observations
		m.mask[:,i] .= 0 
	end
end

function (m::Mill.ArrayModel)(ds::ArrayNode, mask::MatrixMask)
    ArrayNode(m.m(mask.mask .* ds.data))
end

index_in_parent(m::MatrixMask, i) = CartesianIndices((m.rows, m.cols))[i][2]