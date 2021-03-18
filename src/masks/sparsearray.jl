struct SparseArrayMask{M} <: AbstractListMask
	mask::M
	columns::Vector{Int}
end

Flux.@functor(SparseArrayMask)

function Mask(ds::ArrayNode{T,M}, m::ArrayModel, initstats, cluster; verbose::Bool = false) where {T<:SparseMatrixCSC, M}
	nnz(ds.data) == 0 && return(EmptyMask())
	column2cluster = cluster(m, ds)
	columns = identifycolumns(ds.data)
	cluster_assignments = [column2cluster[c] for c in columns]
	if verbose
		n, m = nobs(ds), length(unique(cluster_assignments))
		println("number of sparse vectors: ", n, " number of clusters: ", m, " ratio: ", round(m/n, digits = 3))
	end
	SparseArrayMask(Mask(cluster_assignments, initstats), columns)
end

function Mask(ds::ArrayNode{T,M}, initstats; verbose::Bool = false) where {T<:SparseMatrixCSC, M}
	nnz(ds.data) == 0 && return(EmptyMask())
	columns = identifycolumns(ds.data)
	SparseArrayMask(Mask(nnz(ds.data), initstats), columns)
end

function identifycolumns(x::SparseMatrixCSC)
	columns = findall(!iszero, x);
	columns = [c.I[2] for c in columns]
end

function invalidate!(mask::SparseArrayMask, observations::Vector{Int})
	for (i,c) in enumerate(mask.columns)
		if c ∈ observations
			mask.mask.participate[i] = false
		end
	end
end

function Base.getindex(ds::ArrayNode{T,M}, m::SparseArrayMask, presentobs=fill(true,nobs(ds))) where {T<:Mill.SparseMatrixCSC, M}
	x = deepcopy(ds.data)
	x.nzval[.!m.mask.mask] .= 0
	ArrayNode(x[:,presentobs], ds.metadata)
end

function (m::Mill.ArrayModel)(ds::ArrayNode, mask::SparseArrayMask)
	x = ds.data
	xx = SparseMatrixCSC(x.m, x.n, x.colptr, x.rowval, x.nzval .* mulmask(mask))
    ArrayNode(m.m(xx))
end


index_in_parent(m::SparseArrayMask, i) = m.columns[i]

_nocluster(m::ArrayModel, ds::ArrayNode{T,M})  where {T<:SparseMatrixCSC, M} = unique(identifycolumns(ds.data))