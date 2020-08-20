zerobs() = missing 
emptyexportobs() = Vector{Missing}()


using FillArrays
function OR(xs)
	xs = filter(!ismissing, xs)
	isempty(xs) && return(missing)
	length(xs) > 1 ? LogicalOR(xs) : only(xs)
end
OR(xs::Missing) = missing
struct LogicalOR{T}
	x::T
end
Base.:(==)(e1::LogicalOR, e2::LogicalOR) = e1.x == e2.x
Base.show(io::IO, mime::MIME"text/plain", a::LogicalOR) = println(io, "OR: ",a.x)

# OR(xs) = length(xs) > 1 ? OR(filter(!ismissing, xs)) : only(xs)

"""
	addor(m::Mask, x)

	returns "or" relationships if they are needed in the explanation, by 
	substituing each item with an "or" of items in clusters

	`m` is the mask and 
	`x` is the output of the explanation of bottom layers
"""
function addor(m::Mask{I, D}, x, active) where {I<:Vector{Int}, D}
	xi = m.cluster_membership[active]
	# @show x
	# @show xi
	map(1:length(x)) do i
		ismissing(x[i]) ? missing : OR(x[xi .== xi[i]])
	end
end

addor(m::Mask{I, D}, x::Missing, active)  where {I<: Vector{Int}, D}= missing
addor(m::Mask{I, D}, x, active) where {I<: Nothing, D} = x
addor(m::Mask{I, D}, x::Missing, active) where {I<: Nothing, D}  = missing
addor(m::AbstractExplainMask, x, active) = addor(m.mask, x, active)
addor(m::EmptyMask, x, active) = x


"""
	contributing(m)
	
	returns a mask of items contributing to the explanation
"""
contributing(m::AbstractExplainMask, l) = participate(m) .& prunemask(m)
contributing(m::EmptyMask, l) = Fill(true, l)

function yarason(ds::ArrayNode{T}, m::AbstractExplainMask, e::ExtractCategorical, exportobs = fill(true, nobs(ds))) where {T<:Flux.OneHotMatrix}
	c = contributing(m, nobs(ds))
	items = map(i -> i.ix, ds.data.data)
	!any(c) && return(fill(missing, sum(exportobs)))
	d = reversedict(e.keyvalemap);
	idxs = map(i -> c[i] ? get(d, items[i], "__UNKNOWN__") : missing, findall(exportobs))
	addor(m, idxs, exportobs)
end

"""
	unscale(x,e)

	original values of `x` before the extraction by `e` in json
"""
unscale(x::AbstractArray, e::ExtractScalar) = map(x -> unscale(x,e), x)
unscale(x::Number, e::ExtractScalar) = x / e.s + e.c
unscale(x, e) = x

"""
	yarason(ds, m, e, exportobs::Vector{Bool})

	Values for items in `ds` corresponding to `true` in `prunemask(m)` 
	and `participating(m)`, or `missing` otherwise. Values are exported 
	only for those indicated in binary mask `exportobs`. The export 
	also takes into the account "clusters", which are exported using the 
	`OR` as `OR
"""
function yarason(ds::ArrayNode{T}, m,  e, exportobs = fill(true, nobs(ds))) where {T<:Matrix}
	items = contributing(m, size(ds.data,1))
	x = map(findall(exportobs)) do j 
		[items[i] ? ds.data[i,j] : missing for i in 1:length(items)]
	end
	unscale(x, e)
end

function yarason(ds::ArrayNode{T}, m, e::ExtractString, exportobs = fill(true, nobs(ds))) where {T<:Mill.NGramMatrix}
	c =  contributing(m, nobs(ds))
	x = map(i -> c[i] ? ds.data.s[i] : missing, findall(exportobs))
	addor(m, x, exportobs)
end

function yarason(ds::LazyNode, m, e, exportobs = fill(true, nobs(ds)))
	c =  contributing(m, nobs(ds))
	x = map(i -> c[i] ? ds.data[i] : missing, findall(exportobs))
	addor(m, x, exportobs)
end

# This hack is needed for cases, where an ExtractDict has a single child

# This hack is needed for cases, where scalars are joined to a single matrix
# function yarason(m::Mask, ds::ArrayNode, e::ExtractDict)
# 	ks = keys(e.vec)
# 	s = join(map(i -> "$(i[1]) = $(i[2])" , zip(ks, ds.data[:])))
# 	repr_boolean(:and, unique(s))
# end

function yarason(ds::BagNode, m, e::ExtractArray, exportobs = fill(true, nobs(ds)))
    nobs(ds) == 0 && return(zeroobs())
    ismissing(ds.data) && return(fill(missing, sum(exportobs)))
    !any(exportobs) && emptyexportobs()

    #get indexes of c clusters
	present_childs = Vector(contributing(m, nobs(ds.data)))
	for b in ds.bags[.!exportobs]
	    present_childs[b] .= false
	end

	# @show present_childs
    # !any(exportobs) && return(fill(missing, sum(exportobs)))
    @show sum(present_childs)
	x = yarason(ds.data, m.child, e.item, present_childs)
	@show length(x)
	x = addor(m, x, present_childs)
	bags = Mill.adjustbags(ds.bags, present_childs)[exportobs]
	# @show bags
	map(b -> x[b], bags)
end

# function yarason(ds::BagNode, m, e::JsonGrinder.ExtractKeyAsField)
#     ismissing(ds.data) && return(missing)
#     #get indexes of c clusters
# 	c = contributing(m, nobs(ds.data))
# 	all(.!c) && return(missing)
# 	ss = yarason(ds.data, m.child, e.item)
# 	addor(m, ss, c)
# end


function skipedict(e)
	println("skipedict $(only(keys(e.other)))")
	only(values(e.other))
end

function yarason(ds::BagNode, m, e::ExtractDict{S,V}, exportobs = fill(true, nobs(ds))) where {S<:Nothing, V}
	yarason(ds, m, skipedict(e), exportobs)
end

function yarason(ds::ArrayNode, m, e::ExtractDict{S,V}, exportobs = fill(true, nobs(ds))) where {S<:Nothing, V}
	yarason(ds, m, skipedict(e), exportobs)
end

function yarason(ds::ProductNode{T,M}, m, e::JsonGrinder.ExtractDict, exportobs = fill(true, nobs(ds))) where {T<:NamedTuple, M}
	nobs(ds) == 0 && return(zeroobs())
	!any(exportobs) && emptyexportobs()

	# this awful hack is here if there is a dict of dict with a single key
	if !isnothing(e.other) && isnothing(e.vec) && length(keys(e.other)) == 1 
		ks = keys(ds.data)
		length(ks) > 1 && return(yarason(ds, m, skipedict(e), exportobs))
		only(ks) != only(keys(e.other)) && return(yarason(ds, m, skipedict(e), exportobs))
	end
	#end of hack
	s = map(sort(collect(keys(ds.data)))) do k
		@show (k, e[k])
        k => yarason(ds[k], m[k], e[k], exportobs)
    end
    arrayofdicts(Dict(s), sum(exportobs))
end

function arrayofdicts(d::Dict, l)
	isempty(d) && return(fill(d, l))
	ks = collect(keys(d))
	map(1:l) do i
		Dict(map(k -> k => ismissing(d[k]) ? missing : d[k][i], ks))
	end
end

function yarason(ds::ProductNode{T,M}, m, e::JsonGrinder.ExtractKeyAsField, exportobs = fill(true, nobs(ds))) where {T<:NamedTuple, M}
	nobs(ds) == 0 && return(zeroobs())
	(Dict(
		Symbol(yarason(ds[:key], m[:key],  e.key)) => yarason(ds[:item], m[:item], e.item),
	))
end

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
