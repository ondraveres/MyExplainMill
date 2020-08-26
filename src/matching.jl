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

# # This is just a convenience to iterate over samples ds
# function Base.match(ds::Vector, expression::Dict, extractor::ExtractDict ; path = (), verbose = false)
# 	all(map(x -> match(x, expression, extractor, path = path, verbose = verbose), ds))
# end

# function Base.match(ds::AbstractNode, expression::Vector, extractor; path = (), verbose = false)
# 	all(e -> Base.match(ds, e, extractor; path, verbose), expression)
# end

# absent is always matching
function Base.match(ds::ArrayNode, ::Absent, extractor; path = (), verbose = false)
	printontrue(true, verbose, path," ", "Absent")
end

# matching logical or 
function Base.match(ds, expression::LogicalOR, extractor; path = (), verbose = false) where {T, M}
	any(match(ds, token, extractor;path, verbose) for token in  expression.or)
end


####
#				Dictionary
####
function Base.match(ds::ProductNode, expression::Dict, extractor::ExtractDict{Nothing,V}; path = (), verbose = false) where {V}
	!isempty(setdiff(keys(expression), keys(ds))) && throw("Expression contains keys not in the datasample")
	all(map(k -> match(ds[k], expression[k], extractor[k]; path = (path..., k), verbose = verbose), collect(keys(expression))))
end

function Base.match(ds::ProductNode, expression, extractor::MultipleRepresentation)
	e = extractor.extractors
	ks = collect(keys(e))
	all(match(ds[k], expression, e[k]) for k in ks)
end

####
#				Dictionary with fused Scalar values
####

function Base.match(ds::ProductNode, expression::Dict, extractor::ExtractDict{V,D}; path = (), verbose = false) where {V,D}
	!matcharray(ds[:scalars], expression, extractor.vec) && return(false)
	ks = collect(setdiff(keys(expression), keys(extractor.vec)))
	all(match(ds[k], expression[k], extractor[k]; path = (path..., k), verbose = verbose) for k in ks)
end

function Base.match(ds::ProductNode, expression::Dict, extractor::ExtractDict{V,Nothing}; path = (), verbose = false) where {V}
	match(ds[:scalars], expression, extractor.vec)
end

function matcharray(ds::ArrayNode, vals::Dict, extractors::Dict)
	v = [_getvalue(get(vals, k, absent), f) for (k,f) in extractors]
	active = .!isabsent.(v)
	v = v[active]
	x = ds.data 
	any(view(x,active,i) ≈ v for i in 1:nobs(ds))
end

_getvalue(x::Absent, e) = absent
_getvalue(x, e) = e(x).data[1]

####
#				Scalar
####
function Base.match(ds::ArrayNode, expression::Vector{Vector{T}}, extractor::ExtractScalar; path=(), verbose=false) where {T}
	all(matcharray(ds, v, extractor) for v in expression)
end

function matcharray(ds::ArrayNode, v::Vector, e::ExtractScalar)
	active = .!isabsent.(v)
	v = map(x -> e(x).data[1], v)[active]
	x = ds.data 
	any(view(x,active,i) ≈ v for i in 1:nobs(ds))
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

# function Base.match(ds::BagNode, expression::Vector{Vector{T}}, extractor::JsonGrinder.ExtractKeyAsField; path = (), verbose = false) where {T}
# 	o = map(expression) do ei 
# 		any(match(ds[j].data, ei, extractor.item; path, verbose) for j in 1:nobs(ds))
# 	end
# 	all(o)
# end

# function matchkeyasfield(ds::BagNode, expression, extractor)
# 	any(match(ds[i] for i in 1:nobs(ds)))
# end


function Base.match(ds::ArrayNode, expression::Vector, extractor ; path = (), verbose = false)
	all(match(ds, token, extractor;path, verbose) for token in expression)
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
function Base.match(ds::LazyNode, expression::Vector, extractor ; path = (), verbose = false)
	all(match(ds, token, extractor;path, verbose) for token in expression)
end

function Base.match(ds::LazyNode{T,M}, token::StringOrNum, extractor; path = (), verbose = false) where {T, M}
	printontrue(token ∈ ds.data, verbose, path," ", token)
end

####
#	Matching of 
####

match_or(ds::Vector, e, v; path = (), verbose = false) = any(match(d, e, v; path = path, verbose = verbose) for d in  ds)
