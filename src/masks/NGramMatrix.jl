struct NGramMatrixMask{M} <: AbstractListMask
	mask::M
end

Flux.@functor(NGramMatrixMask)

function Mask(ds::ArrayNode{T,M}, m::ArrayModel, initstats, cluster; verbose = false) where {T<:Mill.NGramMatrix{String}, M}
	cluster_assignments = cluster(m, ds)
	if verbose
		n, m = nobs(ds), length(unique(cluster_assignments)), length(unique(ds.data.s))
		println("number of strings: ", n, " number of clusters: ", m, " ratio: ", round(m/n, digits = 3))
	end
	NGramMatrixMask(Mask(cluster_assignments, initstats))
end

function Mask(ds::ArrayNode{T,M}, initstats; verbose = false) where {T<:Mill.NGramMatrix{String}, M}
	NGramMatrixMask(Mask(nobs(ds.data), initstats))
end

function invalidate!(mask::NGramMatrixMask, observations::Vector{Int})
	participate(mask)[observations] .= false
end

function prune(ds::ArrayNode{T,M}, m::NGramMatrixMask) where {T<:Mill.NGramMatrix{String}, M}
	x = deepcopy(ds.data)
	x.s[.!prunemask(m)] .= ""
	ArrayNode(x, ds.metadata)
end

function Base.getindex(ds::ArrayNode{T,M}, m::NGramMatrixMask, presentobs=fill(true,nobs(ds))) where {T<:Mill.NGramMatrix{String}, M}
	x = deepcopy(ds.data)
	x.s[.!prunemask(m)] .= ""
	ArrayNode(x[presentobs], ds.metadata)
end


function (m::Mill.ArrayModel)(ds::ArrayNode, mask::NGramMatrixMask)
    ArrayNode(m.m(ds.data) .* transpose(mulmask(mask)))
end

_nocluster(m::ArrayModel, ds::ArrayNode{T,M})  where {T<:Mill.NGramMatrix{String}, M} = nobs(ds.data)