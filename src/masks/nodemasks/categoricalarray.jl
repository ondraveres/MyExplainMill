const OneHotNode = ArrayNode{<:Flux.OneHotMatrix, <:Any}

struct CategoricalMask{M} <: AbstractListMask
	mask::M
end

Flux.@functor(CategoricalMask)

function create_mask_structure(ds::OneHotNode, m::ArrayModel, create_mask, cluster)
	cluster_assignments = cluster(m, ds)
	CategoricalMask(create_mask(cluster_assignments))
end

function create_mask_structure(ds::OneHotNode, create_mask)
	CategoricalMask(create_mask(nobs(ds.data)))
end

function Base.getindex(ds::OneHotNode, mk::Union{ObservationMask,CategoricalMask}, presentobs=fill(true, nobs(ds)))
	pm = prunemask(mk.mask)
	ii = map(findall(presentobs)) do j
		i = ds.data.data[j]
		pm[j] ? i.ix : i.of
	end
	x = Flux.onehotbatch(ii, 1:ds.data.height)
	ArrayNode(x, ds.metadata)
end

function invalidate!(mk::CategoricalMask, observations::Vector{Int})
	invalidate!(mk.mask, observations)
end

function present(mk::CategoricalMask, obs)
	map((&), obs, prunemask(mk.mask))
end

function foreach_mask(f, m::CategoricalMask, level, visited)
	if !haskey(visited, m.mask)
		f(m.mask, level)
		visited[m.mask] = nothing
	end
end

function mapmask(f, m::CategoricalMask, level, visited)
	new_mask = get!(visited, m.mask, f(m.mask, level))
	CategoricalMask(new_mask)
end

# This might be actually simplified if we define gradient with respect to ds[mk]
function (m::Mill.ArrayModel)(ds::OneHotNode, mk::Union{ObservationMask,CategoricalMask})
	x = Zygote.@ignore sparse(ds.data)
	y = Zygote.@ignore sparse(fill(size(x)...), collect(1:size(x,2)), 1)
	dm = reshape(diffmask(mk.mask), 1, :)
    m(ArrayNode(@. dm * x + (1 - dm) * y))
end

_nocluster(m::ArrayModel, ds::OneHotNode) = nobs(ds.data)