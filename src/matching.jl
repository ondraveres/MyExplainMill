import Base.match

# The idea is to extend the Base.match, such that we can check, if 
# explanations (input to YaraGens) are correct

function printontrue(o, verbose,  s...)
	verbose && o && println(s)
	o 
end

function Base.match(d::Dict, e, v; path = (), verbose = false)
	ks = collect(keys(d))
	if length(ks) == 1 
		k = only(ks)
		k == :or && return(match_or(d[:or], e, v; path = path, verbose = verbose))
		k == :and && return(match_and(d[:and], e, v; path = path, verbose = verbose))
	end
	match_product(d, e, v; path = path, verbose = verbose)
end

function match_product(d::Dict, e, v::ProductNode; path = (), verbose = false)
	ks = intersect(keys(d), keys(v))
	all(map(k -> match(d[k], e[k], v[k]; path = (path..., k), verbose = verbose), collect(ks)))
end

function Base.match(s::String, e, v::BagNode; path = (), verbose = false)
	match(s, e.item, v.data; path = (path..., :item), verbose = verbose)
end

function Base.match(s::String, e, v::ArrayNode{T,N}; path = (), verbose = false) where {T<: NGramMatrix, N}
	printontrue(s ∈ v.data.s, verbose, path," ", s)
end

function Base.match(s::String, e::ExtractCategorical, v::ArrayNode{T,N}; path = (), verbose = false) where {T<:Flux.OneHotMatrix, N}
	idxs = map(i -> i.ix, v.data.data)
	printontrue(e.keyvalemap[s] ∈ idxs, verbose, path," ", s)
end

match_and(ds::Vector, e, v; path = (), verbose = false) = all(match(d, e, v; path = path, verbose = verbose) for d in  ds)
match_or(ds::Vector, e, v; path = (), verbose = false) = any(match(d, e, v; path = path, verbose = verbose) for d in  ds)
