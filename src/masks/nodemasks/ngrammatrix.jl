struct NGramMatrixMask{M} <: AbstractListMask
	mask::M
end

Flux.@functor(NGramMatrixMask)

function create_mask_structure(ds::ArrayNode{T,M}, m::ArrayModel, create_mask, cluster) where {T<:Mill.NGramMatrix{String}, M}
	cluster_assignments = cluster(m, ds)
	NGramMatrixMask(create_mask(cluster_assignments))
end

function create_mask_structure(ds::ArrayNode{T,M}, initstats) where {T<:Mill.NGramMatrix{String}, M}
	NGramMatrixMask(create_mask(nobs(ds.data)))
end

function invalidate!(mask::NGramMatrixMask, observations::Vector{Int})
	participate(mask)[observations] .= false
end

function Base.getindex(ds::ArrayNode{T,M}, mk::NGramMatrixMask, presentobs=fill(true,nobs(ds))) where {T<:Mill.NGramMatrix{String}, M}
	x = ds.data
	pm = prunemask(mk.mask) 
	s = map(findall(presentobs)) do i 
		pm[i] ? x.s[i] : ""
	end
	ArrayNode(NGramMatrix(s, x.n, x.b, x.m), ds.metadata)
end


function (m::Mill.ArrayModel)(ds::ArrayNode, mk::NGramMatrixMask)
    ArrayNode(m.m(ds.data) .* transpose(diffmask(mk.mask)))
end

_nocluster(m::ArrayModel, ds::ArrayNode{T,M})  where {T<:Mill.NGramMatrix{String}, M} = nobs(ds.data)