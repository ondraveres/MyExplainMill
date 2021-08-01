struct SparseArrayMask{M} <: AbstractListMask
	mask::M
	columns::Vector{Int}
end

const SparseNode = ArrayNode{<:SparseMatrixCSC, <:Any}

Flux.@functor(SparseArrayMask)

function create_mask_structure(ds::SparseNode, m::ArrayModel, create_mask, cluster)
	nnz(ds.data) == 0 && return(EmptyMask())
	column2cluster = cluster(m, ds)
	columns = identifycolumns(ds.data)
	cluster_assignments = [column2cluster[c] for c in columns]
	SparseArrayMask(create_mask(cluster_assignments), columns)
end

function create_mask_structure(ds::SparseNode, create_mask)
	nnz(ds.data) == 0 && return(EmptyMask())
	columns = identifycolumns(ds.data)
	SparseArrayMask(create_mask(nnz(ds.data)), columns)
end

function identifycolumns(x::SparseMatrixCSC)
	columns = findall(!iszero, x);
	columns = [c.I[2] for c in columns]
end

function invalidate!(mk::SparseArrayMask, observations::Vector{Int})
	for (i,c) in enumerate(mk.columns)
		if c ∈ observations
			invalidate!(mk.mask, i)
		end
	end
end

function Base.getindex(ds::SparseNode, mk::SparseArrayMask, presentobs=fill(true,nobs(ds)))
	x = deepcopy(ds.data)
	x.nzval[.!prunemask(mk.mask)] .= 0
	ArrayNode(x[:,presentobs], ds.metadata)
end

function Base.getindex(ds::SparseNode, mk::ObservationMask, presentobs=fill(true,nobs(ds)))
	x = deepcopy(ds.data)
	for (i, m) in enumerate(prunemask(mk.mask))
		m && continue
		x.nzval[x.colptr[i]:x.colptr[i+1]-1] .= 0
	end
	ArrayNode(x[:,presentobs], ds.metadata)
end

function present(mk::SparseArrayMask, obs)
	a = fill(false, length(obs))
	for (i, b) in enumerate(prunemask(mk.mask))
		a[mk.columns[i]] |= b
	end
	map((&), a, obs)
end

function foreach_mask(f, m::SparseArrayMask, level = 1)
	f(m.mask, level)
end

function mapmask(f, m::SparseArrayMask, level = 1)
	SparseArrayMask(f(m.mask, level), m.columns)
end

function (m::Mill.ArrayModel)(ds::SparseNode, mk::SparseArrayMask)
	x = ds.data
	nzval = x.nzval .* diffmask(mk.mask)
	xx = dense_sparse(x.m, x.n, x.colptr, x.rowval, nzval)
    ArrayNode(m.m(xx))
end

function (m::Mill.ArrayModel)(ds::SparseNode, mk::ObservationMask)
	xx = ds.data .* transpose(diffmask(mk.mask))
    ArrayNode(m.m(xx))
end

dense_sparse(m::Int, n::Int, colptr, rowval, nzval) = Matrix(SparseMatrixCSC(m, n, colptr, rowval, nzval))

Zygote.@adjoint function dense_sparse(m::Int, n::Int, colptr, rowval, nzval)
	∂f = Δ -> begin
		∇nzval = similar(nzval)
		iₙ = 1 
		for ci in 1:n
			for j in colptr[ci]:(colptr[ci+1] - 1)
				ri = rowval[j]
				∇nzval[iₙ] = Δ[ri,ci]
				iₙ += 1
			end
		end
		(nothing, nothing, nothing, nothing, ∇nzval)
	end
	Matrix(SparseMatrixCSC(m, n, colptr, rowval, nzval)), ∂f
end

_nocluster(m::ArrayModel, ds::SparseNode) = unique(identifycolumns(ds.data))