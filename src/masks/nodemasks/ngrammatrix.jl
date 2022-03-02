struct NGramMatrixMask{M} <: AbstractListMask
	mask::M
end

const NGramNode = ArrayNode{<:Mill.NGramMatrix{String},<:Any}

Flux.@functor(NGramMatrixMask)

function create_mask_structure(ds::NGramNode, m::ArrayModel, create_mask, cluster)
	cluster_assignments = cluster(m, ds)
	NGramMatrixMask(create_mask(cluster_assignments))
end

function create_mask_structure(ds::NGramNode, create_mask)
	NGramMatrixMask(create_mask(nobs(ds.data)))
end

function invalidate!(mk::NGramMatrixMask, invalid_observations::AbstractVector{Int})
	invalidate!(mk.mask, invalid_observations)
end

function Base.getindex(ds::NGramNode, mk::Union{ObservationMask,NGramMatrixMask}, presentobs=fill(true,nobs(ds)))
	x = ds.data
	pm = prunemask(mk.mask) 
	s = map(findall(presentobs)) do i 
		pm[i] ? x.S[i] : ""
	end
	ArrayNode(NGramMatrix(s, x.n, x.b, x.m), ds.metadata)
end

function present(mk::NGramMatrixMask, obs)
	map((&), obs, prunemask(mk.mask))
end

function foreach_mask(f, m::NGramMatrixMask, level, visited)
	if !haskey(visited, m.mask)
		f(m.mask, level)
		visited[m.mask] = nothing
	end
end

function mapmask(f, m::NGramMatrixMask, level, visited)
	new_mask = get!(visited, m.mask, f(m.mask, level))
	NGramMatrixMask(new_mask)
end

	
# TODO: We should make this one faster by writing a custom gradient for interpolation
# since we do not need gradients with respect to `x` and `y`
function (m::Mill.ArrayModel)(ds::NGramNode, mk::Union{ObservationMask,NGramMatrixMask})
	ng = ds.data
	eg = NGramMatrix(fill("", length(ng.S)), ng.n, ng.b, ng.m)
	x = Zygote.@ignore Matrix(SparseMatrixCSC{Float32, Int64}(ng))
	y = Zygote.@ignore Matrix(SparseMatrixCSC{Float32, Int64}(eg))
	dm = reshape(diffmask(mk.mask), 1, :)
	x′ = @. dm * x + (1 - dm) * y
    m(ArrayNode(x′))
end

_nocluster(m::ArrayModel, ds::NGramNode)  = nobs(ds.data)