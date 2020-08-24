import Base.match

const StringOrNum = Union{Number, AbstractString}
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

function Base.match(ds::AbstractNode, expression::Vector, extractor; path = (), verbose = false)
	all(e -> Base.match(ds, e, extractor; path, verbose), expression)
end

# missing is always matching
function Base.match(ds::ArrayNode, ::Missing, extractor; path = (), verbose = false)
	printontrue(true, verbose, path," ", "Missing")
end

####
#				Matching of Bags
####
function Base.match(ds::BagNode, expression::Vector, extractor::ExtractArray ; path = (), verbose = false)
	all(matchbag(ds, e, extractor;path, verbose) for e in expression)
end

function matchbag(ds::BagNode, expression, extractor::ExtractArray; path = (), verbose = false)
	o = map(expression) do ei 
		any(match(ds[j].data, ei, extractor.item; path, verbose) for j in 1:nobs(ds))
	end
	all(o)
end

####
#				Matching of Strings
####
function Base.match(ds::ArrayNode{T,M}, token::AbstractString, extractor::ExtractString; path = (), verbose = false) where {T<: NGramMatrix, M}
	printontrue(token ∈ ds.data.s, verbose, path," ", token)
end


####
#				Matching of Categorical
####
function Base.match(ds::ArrayNode{T,M}, token::StringOrNum, extractor::ExtractCategorical; path = (), verbose = false) where {T<: Flux.OneHotMatrix, M}
	idxs = map(i -> i.ix, unique(ds.data.data))
	i = get(extractor.keyvalemap, token, ds.data.height)
	printontrue(i ∈ idxs, verbose, path," ", token)
end

####
#				Matching of Lazy
####
function Base.match(ds::LazyNode{T,M}, token::StringOrNum, extractor; path = (), verbose = false) where {T, M}
	printontrue(token ∈ ds.data, verbose, path," ", token)
end

match_or(ds::Vector, e, v; path = (), verbose = false) = any(match(d, e, v; path = path, verbose = verbose) for d in  ds)
