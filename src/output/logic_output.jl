import Base.repr
child_mask(m::BagMask) = m.child
function child_mask(m::BagMask{C,B}) where {C<:EmptyMask, B}
	m.mask
end

function child_mask(m::EmptyMask)
	@info "How did I get to here"
	EmptyMask()
end

function islogical(s::Dict{Symbol,T}) where {T}
	collect(keys(s)) == [:and] && return(true)
	collect(keys(s)) == [:or] && return(true)
	return(false)
end

islogical(s::Vector) = s

sortunique(s::Vector{T}) where {T<:AbstractString} = sort(unique(s))
sortunique(s::Vector{T}) where {T<:Number} = sort(unique(s))
sortunique(s::Vector{T}) where {T<:Dict} = unique(s)
function sortunique(s::Vector)
	if mapreduce(typeof, promote_type, s) <:AbstractString
		return(sortunique(String.(s)))
	else 
		return(unique(s))
	end
end

function repr_boolean(op::Symbol, s::Vector{T}; thin::Bool = true) where {T}
	s = filter(!isempty, s)
	s = sortunique(unique(s))
	if isempty(s) 
		return(Dict{Symbol,T}())
	elseif length(s) == 1
		if thin 
			return(only(s))
		else 
			return(Dict(op => s))
		end
	else
		return(Dict(op => s))
	end
end

function repr_boolean(op::Symbol, s::Dict{Symbol,String}; thin::Bool = true) 
	if thin || isempty(s)
		return(s)
	elseif islogical(s)
		return(s)
	else 
		return(Dict(op => s))
	end
end

isarray(s::String) = false
isarray(s::Vector) = true
function isarray(d::Dict)
	ks = keys(d)
	if length(ks) == 1
		k = only(ks)
		k == :or && return(false)
		k == :and && return(isarray(d[k]))
	end
	return(false)
end

function repr(mim::MIME"text/json", m::AbstractExplainMask, ds::ArrayNode{T}, e::E) where {T<:Flux.OneHotMatrix, E<:ExtractDict}
	length(e.other) > 1 &&  @error "This should not happen"
	k = collect(keys(e.other))[1]
	repr(mim, m, ds, e.other[k])
end

function repr(::MIME"text/json", m::AbstractExplainMask, ds::ArrayNode{T}, e) where {T<:Flux.OneHotMatrix}
	contributing = participate(m) .& prunemask(m)
	items = map(i -> i.ix, ds.data.data)
	items = items[contributing]
	isempty(items) && return(Dict{Symbol, String}())

	idxs = unique(items);
	d = reversedict(e.keyvalemap);
	idxs = filter(i -> haskey(d, i), idxs)
	isempty(idxs) && Dict{Symbol, String}()
	s = map(i -> d[i], idxs)
	repr_boolean(:and, s)
end

function repr(::MIME"text/json", m::AbstractExplainMask, ds::ArrayNode{T}, e) where {T<:Matrix}
	items = participate(m) .& prunemask(m)
	items = findall(items)
	items = filter(i -> any(ds.data[i,:] .!= 0), items)
	isempty(items) && return(Dict{Symbol, String}())
	repr_boolean(:and, Dict([Symbol(i) => "["*join(ds.data[i,:],",")*"]" for i in items]))
end

function repr(::MIME"text/json", m::EmptyMask, ds::ArrayNode{T}, e) where {T<:Matrix}
	return(Dict{Symbol,String}())
end

# function repr(::MIME"text/json":SparseArrayMask, ds::ArrayNode{T}, e) where {T<:Mill.NGramMatrix}
function repr(::MIME"text/json", m::AbstractExplainMask, ds::ArrayNode{T}, e) where {T<:Mill.NGramMatrix}
	repr_boolean(:and, ds.data.s[participate(m) .& prunemask(m)])
end

function repr(::MIME"text/json", m::EmptyMask, ds::ArrayNode{T}, e) where {T<:Mill.NGramMatrix}
	repr_boolean(:and, unique(ds.data.s))
end

function repr(::MIME"text/json", m::Mask, ds::ArrayNode, e::ExtractDict)
	ks = keys(e.vec)
	s = join(map(i -> "$(i[1]) = $(i[2])" , zip(ks, ds.data[:])))
	repr_boolean(:and, unique(s))
end


function repr(mim::MIME"text/json", m::AbstractExplainMask, ds::BagNode, e)
    ismissing(ds.data) && return(Dict{Symbol, String}())

    #get indexes of contributing clusters
	contributing_indexes = participate(m) .& prunemask(m)
	all(.!contributing_indexes) && return(Dict{Symbol, String}())
	if isnothing(m.mask.cluster_membership)
		ss = repr(mim, child_mask(m), ds.data, e.item)
		s = isarray(ss) ? ss : [ss]
		return(s)
	else
		cluster_membership = m.mask.cluster_membership
		contributing_clusters = unique(cluster_membership[contributing_indexes])
		cluster2instance = idmap(cluster_membership)

		#now we need to express the OR relationship within clusters 
		#and AND relationship across clusters
		ss = map(contributing_clusters) do k
			ss = map(cluster2instance[k]) do i
				mm = deepcopy(child_mask(m))
				invalidate!(mm,setdiff(1:nobs(ds.data), i))
				repr(mim, mm, ds.data, e.item)
			end
			repr_boolean(:or, ss)
		end
		ss = repr_boolean(:and, ss)
		ss = isarray(ss) ? ss : [ss]
		return(ss)
	end
end


function repr(mim::MIME"text/json", m::AbstractExplainMask, ds::BagNode, e::JsonGrinder.ExtractKeyAsField)
    ismissing(ds.data) && return(Dict{Symbol, String}())

    #get indexes of contributing clusters
	contributing_indexes = participate(m) .& prunemask(m)
	all(.!contributing_indexes) && return(Dict{Symbol, String}())
	if isnothing(m.mask.cluster_membership)
		ss = repr(mim, child_mask(m), ds.data, e)
		s = isarray(ss) ? ss : [ss]
		return(s)
	else
		error("this is not implemented")
		return(ss)
	end
end

repr(mim::MIME"text/json", m::EmptyMask, ds::BagNode, e::JsonGrinder.ExtractKeyAsField) = Dict{Symbol,String}()

function repr(mim::MIME"text/json", m::EmptyMask, ds::BagNode, e)
    ismissing(ds.data) && return(Dict{Symbol,String}())
    nobs(ds.data) == 0 && return(Dict{Symbol,String}())
    ss = repr(mim, m, ds.data, e.item);
    repr_boolean(:and, ss, thin = false)
end

function repr(mim::MIME"text/json", m::AbstractExplainMask, ds::ProductNode{T,M}, e) where {T<:NamedTuple, M}
	nobs(ds) == 0 && return(Dict{Symbol,String}())
	s = map(sort(collect(keys(ds.data)))) do k
        subs = repr(mim, m[k], ds[k], e[k])
        isempty(subs) ? nothing : k => subs
    end
    Dict(filter(!isnothing, s))
end
function repr(mim::MIME"text/json", m::AbstractExplainMask, ds::ProductNode{T,M}, e::JsonGrinder.ExtractKeyAsField) where {T<:NamedTuple, M}
	nobs(ds) == 0 && return(Dict{Symbol,String}())
	(Dict(
		Symbol(repr(mim, m[:key], ds[:key], e.key)) => repr(mim, m[:item], ds[:item], e.item),
	))
end

function repr(mim::MIME"text/json", m::AbstractExplainMask, ds::ProductNode{T,M}, e::MultipleRepresentation) where {T<:Tuple, M}
	nobs(ds) == 0 && return(Dict{Symbol,String}())
	s = map(sort(collect(keys(ds.data)))) do k
        subs = repr(mim, m.childs[k], ds.data[k], e.extractors[k])
        isempty(subs) ? nothing : k => subs
    end
    Dict(filter(!isnothing, s))
end

function e2boolean(pruning_mask, dss, extractor)
	d = map(1:nobs(dss)) do i 
		mapmask(pruning_mask) do m 
			participate(m) .= true
		end
		invalidate!(pruning_mask,setdiff(1:nobs(dss), i))
		repr(MIME("text/json"),pruning_mask, dss, extractor);
	end
	d = filter(!isempty, d)
	isempty(d) && return([Dict{Symbol,Any}()])
	d = unique(d)
	d = length(d) > 1 ? ExplainMill.repr_boolean(:or, d) : d
end
