OR(xs) = length(xs) > 1 ? Dict(:or => filter(!ismissing, xs)) : xs

"""
	addor(m::Mask, x)

	returns "or" relationships if they are needed in the explanation, by 
	substituing each item with an "or" of items in clusters

	`m` is the mask and 
	`x` is the output of the explanation of bottom layers
"""
addor(m, x) = addor(m, x, participate(m) .& prunemask(m))

function addor(m::Mask{I, D}, x, active) where {I<:Vector{Int}, D}
	xi = m.cluster_membership[active]
	map(1:length(x)) do i
		ismissing(x[i]) ? missing : OR(x[xi .== xi[i]])
	end
end

addor(m::Mask{I, D}, x::Missing, active)  where {I<: Vector{Int}, D}= missing
addor(m::Mask{I, D}, x, active) where {I<: Nothing, D} = x
addor(m::Mask{I, D}, x::Missing, active) where {I<: Nothing, D}  = missing

# below is the version, where "unknown keys are not exported"
# function yarason(ds::ArrayNode{T}, m::AbstractExplainMask, e::ExtractCategorical) where {T<:Flux.OneHotMatrix}
# 	contributing = participate(m) .& prunemask(m)
# 	items = map(i -> i.ix, ds.data.data)
# 	d = reversedict(e.keyvalemap);
# 	contributing = contributing .& map(i -> haskey(d, i), items)
# 	!any(contributing) && return(missing)
# 	idxs = map(i -> d[i], items[contributing])
# 	addor(m.mask, idxs, contributing)
# end

function yarason(ds::ArrayNode{T}, m::AbstractExplainMask, e::ExtractCategorical, exportobs = fill(true, nobs(ds))) where {T<:Flux.OneHotMatrix}
	contributing = participate(m) .& prunemask(m)
	items = map(i -> i.ix, ds.data.data)
	!any(contributing) && return(fill(missing, sum(exportobs)))
	d = reversedict(e.keyvalemap);
	idxs = map(i -> contributing[i] ? get(d, items[i], "__UNKNOWN__") : missing, findall(exportobs))
	addor(m.mask, idxs, exportobs)
end

# function yarason(m::AbstractExplainMask, ds::ArrayNode{T}, e) where {T<:Matrix}
# 	items = participate(m) .& prunemask(m)
# 	items = findall(items)
# 	items = filter(i -> any(ds.data[i,:] .!= 0), items)
# 	isempty(items) && return(missing)
# 	Dict([Symbol(i) => "["*join(ds.data[i,:],",")*"]" for i in items])
# end

function yarason(ds::ArrayNode{T}, m::AbstractExplainMask, e, exportobs = fill(true, nobs(ds))) where {T<:Mill.NGramMatrix}
	contributing =  participate(m) .& prunemask(m)
	x = map(i -> contributing[i] ? ds.data.s[i] : missing, exportobs)
	addor(m.mask, x, exportobs)
end

function yarason(ds::ArrayNode{T}, m::EmptyMask, e, exportobs = fill(true, nobs(ds))) where {T<:Mill.NGramMatrix}
	ds.data.s[exportobs]
end

function yarason(ds::LazyNode, m::AbstractExplainMask, e, exportobs = fill(true, nobs(ds)))
	contributing =  participate(m) .& prunemask(m)
	x = map(i -> contributing[i] ? ds.data[i] : missing, exportobs)
	addor(m.mask, x, exportobs)
end

function yarason(ds::LazyNode, m::EmptyMask, e, exportobs = fill(true, nobs(ds)))
	ds.data[exportobs]
end

# function yarason(m::Mask, ds::ArrayNode, e::ExtractDict)
# 	ks = keys(e.vec)
# 	s = join(map(i -> "$(i[1]) = $(i[2])" , zip(ks, ds.data[:])))
# 	repr_boolean(:and, unique(s))
# end

function yarason(ds::BagNode, m::AbstractExplainMask, e::ExtractArray)
    ismissing(ds.data) && return(missing)

    #get indexes of contributing clusters
	contributing = participate(m) .& prunemask(m)
	!any(contributing) && return(missing)
	ss = yarason(ds.data, m.child, e.item)
	addor(m.mask, ss, contributing)
end

function yarason(ds::BagNode, m::AbstractExplainMask, e::JsonGrinder.ExtractKeyAsField)
    ismissing(ds.data) && return(missing)

    #get indexes of contributing clusters
	contributing = participate(m) .& prunemask(m)
	all(.!contributing) && return(missing)
	ss = yarason(ds.data, m.child, e.item)
	addor(m.mask, ss, contributing)
end

yarason(ds::BagNode, m::EmptyMask, e::JsonGrinder.ExtractKeyAsField) = missing

function yarason(m::EmptyMask, ds::BagNode, e)
    ismissing(ds.data) && return(missing)
    nobs(ds.data) == 0 && return(missing)
    yarason(ds.data, m, e.item);
end

function yarason(ds::ProductNode{T,M}, m::AbstractExplainMask, e) where {T<:NamedTuple, M}
	nobs(ds) == 0 && return(missing)
	s = map(sort(collect(keys(ds.data)))) do k
        subs = yarason(m[k], ds[k], e[k])
        isempty(subs) ? nothing : k => subs
    end
    Dict(filter(!isnothing, s))
end

function yarason(ds::ProductNode{T,M}, m::AbstractExplainMask, e::JsonGrinder.ExtractKeyAsField) where {T<:NamedTuple, M}
	nobs(ds) == 0 && return(missing)
	(Dict(
		Symbol(yarason(ds[:key], m[:key],  e.key)) => yarason(ds[:item], m[:item], e.item),
	))
end

# function yarason(ds::ProductNode{T,M}, m::AbstractExplainMask, e::MultipleRepresentation) where {T<:Tuple, M}
# 	nobs(ds) == 0 && return(missing)
# 	s = map(sort(collect(keys(ds.data)))) do k
#         subs = yarason(m.childs[k], ds.data[k], e.extractors[k])
#         isempty(subs) ? nothing : k => subs
#     end
#     Dict(filter(!isnothing, s))
# end

# function e2boolean(pruning_mask, dss, extractor)
# 	d = map(1:nobs(dss)) do i 
# 		mapmask(pruning_mask) do m 
# 			participate(m) .= true
# 		end
# 		invalidate!(pruning_mask,setdiff(1:nobs(dss), i))
# 		repr(MIME("text/json"),pruning_mask, dss, extractor);
# 	end
# 	d = filter(!isempty, d)
# 	isempty(d) && return([Dict{Symbol,Any}()])
# 	d = unique(d)
# 	d = length(d) > 1 ? ExplainMill.repr_boolean(:or, d) : d
# end
