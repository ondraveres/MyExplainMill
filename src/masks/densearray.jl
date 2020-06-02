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

# Mask(ds::ArrayNode{T,M}, m::ArrayModel, initstats, cluster; verbose::Bool = false) where {T<:Matrix, M}  = EmptyMask()

function Mask(ds::ArrayNode{T,M}, m::ArrayModel, initstats, cluster; verbose::Bool = false) where {T<:Matrix, M} 
	# Mask(ds, initstats, verbose = verbose)
	MatrixMask(Mask(size(ds.data, 1), initstats), size(ds.data)...)
end

function Mask(ds::ArrayNode{T,M}, initstats; verbose::Bool = false) where {T<:Matrix, M} 
	# MatrixMask(Mask(length(ds.data), initstats), size(ds.data)...)
	MatrixMask(Mask(size(ds.data, 1), initstats), size(ds.data)...)
end

function prune(ds::ArrayNode{T,M}, m::MatrixMask) where {T<:Matrix, M}
	x = deepcopy(ds.data)
	x[.!prunemask(m), :] .= 0
	ArrayNode(x, ds.metadata)
end

function invalidate!(mask::MatrixMask, observations::Vector{Int})
end

# function invalidate!(m::MatrixMask, observations::Vector{Int})
# 	@show m.mask.mask
# 	for i in observations
# 		m.mask.mask[:,i] .= 0 
# 	end
# end

function (m::Mill.ArrayModel)(ds::ArrayNode, mask::MatrixMask)
    ArrayNode(m.m(mulmask(mask) .* ds.data))
end

index_in_parent(m::MatrixMask, i) = CartesianIndices((m.rows, m.cols))[i][2]

_nocluster(m::ArrayModel, ds::ArrayNode{T,M}) where {T<:Matrix,M} = size(ds.data, 1)