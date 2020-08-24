import Base.match

# The idea is to extend the Base.match, such that we can check, if 
# explanations (input to YaraGens) are correct

function printontrue(o, verbose,  s...)
	verbose && o && println(s)
	o 
end

function Base.match(d::Dict, e, v; path = (), verbose = false)
	isempty(d) && return(false)
	ks = collect(keys(d))
	match_product(d, e, v; path = path, verbose = verbose)
end

# This is just a convenience to iterate over samples ds
function Base.match(ds::Vector, expression::Dict, extractor::ExtractDict ; path = (), verbose = false)
	all(map(x -> match(x, expression, extractor, path = path, verbose = verbose), ds))
end

# This matches product / dictionary / product
function Base.match(ds::ProductNode, expression::Dict, extractor::ExtractDict; path = (), verbose = false)
	ks = intersect(keys(ds), keys(expression))
	all(map(k -> match(ds[k], expression[k], extractor[k]; path = (path..., k), verbose = verbose), collect(ks)))
end

# here, we need to evaluate that every item in the expression is in the bag
function Base.match(ds::BagNode, expression::Vector, extractor::ExtractArray ; path = (), verbose = false)
	o = map(expression) do ei 
		any(match(data[j], ei, extractor.item; path, verbose) for j in 1:nobs(ds.data))
	end
	all(o)
end

function Base.match(ds::ArrayNode, expression::Vector, extractor; path = (), verbose = false)
	all(e -> Base.match(ds, e, extractor; path, verbose), expression)
end

function Base.match(ds::ArrayNode{T,M}, expression::S, extractor::ExtractString; path = (), verbose = false) where {T<: NGramMatrix, M, S<:String}
	printontrue(expression ∈ ds.data.s, verbose, path," ", expression)
end

function Base.match(ds::ArrayNode, expression::Missing, extractor; path = (), verbose = false)
	printontrue(true, verbose, path," ", "Missing")
end


function Base.match(s::Vector, e, v::ArrayNode{T,N}; path = (), verbose = false) where {T<: NGramMatrix, N}
	all(x -> printontrue(x ∈ v.data.s, verbose, path," ", x), s)
end

function Base.match(s::String, e::ExtractCategorical, v::ArrayNode{T,M}; path = (), verbose = false) where {T<:Flux.OneHotMatrix, M}
	idxs = map(i -> i.ix, v.data.data)
	printontrue(e.keyvalemap[s] ∈ idxs, verbose, path," ", s)
end

function Base.match(s::Vector, e::ExtractCategorical, v::ArrayNode{T,N}; path = (), verbose = false) where {T<:Flux.OneHotMatrix, N}
	all(map(x -> match(x, e, v, path = path, verbose = verbose), s))
end

match_or(ds::Vector, e, v; path = (), verbose = false) = any(match(d, e, v; path = path, verbose = verbose) for d in  ds)
