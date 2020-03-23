struct SparseArrayMask <: AbstractListMask
	mask::Mask
	columns::Vector{Int}
end

function Mask(ds::ArrayNode{T,M}) where {T<:SparseMatrixCSC, M}
	nnz(ds.data) == 0 && return(EmptyMask())
	columns = findall(!iszero, ds.data);
	columns = [c.I[2] for c in columns]
	SparseArrayMask(Mask(length(columns)), columns)
end

function Mask(ds::ArrayNode{T,M}, m::ArrayModel; cluster_algorithm = cluster_instances, verbose::Bool = false) where {T<:SparseMatrixCSC, M}
	nnz(ds.data) == 0 && return(EmptyMask())
	column2cluster = cluster_algorithm(m(ds).data)
	columns = findall(!iszero, ds.data);
	columns = [c.I[2] for c in columns]
	cluster_assignments = [column2cluster[c] for c in columns]
	if verbose
		n, m = nobs(ds), length(unique(cluster_assignments))
		println("number of sparse vectors: ", n, " number of clusters: ", m, " ratio: ", round(m/n, digits = 3))
	end
	SparseArrayMask(Mask(cluster_assignments), columns)
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
