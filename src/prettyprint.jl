using JsonGrinder
import Base.repr

reversedict(d) = Dict([v => k for (k,v) in d]...)

# copied from Mill.jl, which now does not have it
const COLORS = [:blue, :red, :green, :yellow, :cyan, :magenta]

function paddedprint(io, s...; color=:default, pad=[])
    for (c, p) in pad
        printstyled(io, p, color=c)
    end
    printstyled(io, s..., color=color)
end

function print_explained(io, ds::ArrayNode{T}, e::E; pad = []) where {T<:Flux.OneHotMatrix, E<:ExtractCategorical}
	c = COLORS[(length(pad)%length(COLORS))+1]
	x = ds.data
	idxs = map(argmax, x.data)
	filter!(i -> x.height != i, idxs)
	if isempty(idxs)
		s = "∅"
		paddedprint(io, s, color = c)
	else
		idxs = unique(idxs);
		d = reversedict(e.keyvalemap);
		s = join(map(i -> d[i], idxs), " and ")
		paddedprint(io, s, color = c)
	end
end

function print_explained(io, ds::ArrayNode{T}, e::E; pad = []) where {T<:Flux.OneHotMatrix, E<:ExtractBranch}
	length(e.other) > 1 &&  @error "This should not happen"
	k =collect(keys(e.other))[1]
	print_explained(io, ds, e.other[k])
end

function print_explained(io, ds::ArrayNode{T}, e; pad = []) where {T<:Mill.NGramMatrix}
    c = COLORS[(length(pad)%length(COLORS))+1]
	s = filter(!isempty, ds.data.s)
	if isempty(s)
		s = "∅"
		paddedprint(io, s, color = c)
	else
		s = join(s,", ")
		paddedprint(io, s, color = c)
	end
end

function print_explained(io, ds::BagNode, e; pad = [])
    c = COLORS[(length(pad)%length(COLORS))+1]
    if ismissing(ds.data)
	    paddedprint(io,"∅", color = c)
	else
	    paddedprint(io,"List of\n", color=c)
	    paddedprint(io, "    └── ", color=c, pad=pad)
	    print_explained(io, ds.data, e.item, pad = [pad; (c, "      ")])
	end
end

function print_explained(io::IO, n::AbstractProductNode, e; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "ProductNode", color=c)
    m = length(n.data)
    ks = sort(collect(keys(n.data)))
    for i in 1:(m-1)
        println(io)
        paddedprint(io, "  ├── $(ks[i]): ", color=c, pad=pad)
        print_explained(io, n[ks[i]], e[ks[i]], pad=[pad; (c, "  │" * repeat(" ", max(3, 2+length(String(ks[i])))))])
    end
    println(io)
    paddedprint(io, "  └── $(ks[end]): ", color=c, pad=pad)
    print_explained(io, n[ks[end]], e[ks[end]], pad=[pad; (c, repeat(" ", 3+max(3, 2+length(String(ks[end])))))])
end

print_explained(ds::AbstractNode, e) = print_explained(stdout, ds, e)


child_mask(m::BagMask) = m.child
function child_mask(m::BagMask{C,B}) where {C<:EmptyMask, B}
	m.mask
end
function child_mask(m::EmptyMask)
	@info "How did I get to here"
	EmptyMask()
end


function repr_boolean(op::Symbol, s::Vector{T}) where {T}
	s = filter(!isempty, s)
	s = unique(s)
	if isempty(s) 
		return(Dict{Symbol,T}())
	elseif length(s) == 1
		return(only(s))
	else
		return(Dict(op => s))
	end
end

repr_boolean(::Symbol, d::Dict{Symbol,String}) = d

# function print_explained(io, m::CategoricalMask, ds::ArrayNode{T}, e::E; pad = []) where {T<:Flux.OneHotMatrix, E<:ExtractBranch}
function repr(mim::MIME"text/json", m::AbstractExplainMask, ds::ArrayNode{T}, e::E) where {T<:Flux.OneHotMatrix, E<:ExtractBranch}
	length(e.other) > 1 &&  @error "This should not happen"
	k = collect(keys(e.other))[1]
	repr(mim, m, ds, e.other[k])
end

# function repr(::MIME"text/json":CategoricalMask, ds::ArrayNode{T}, e::E) where {T<:Flux.OneHotMatrix, E<:ExtractCategorical}
function repr(::MIME"text/json", m::AbstractExplainMask, ds::ArrayNode{T}, e) where {T<:Flux.OneHotMatrix}
	contributing = participate(m) .& mask(m)
	items = map(i -> i.ix, ds.data.data)
	items = items[contributing]
	isempty(items) && Dict{Symbol, String}()

	idxs = unique(items);
	d = reversedict(e.keyvalemap);
	s = map(i -> d[i], idxs)
	repr_boolean(:and, s)
end

# function repr(::MIME"text/json":SparseArrayMask, ds::ArrayNode{T}, e) where {T<:Mill.NGramMatrix}
function repr(::MIME"text/json", m::AbstractExplainMask, ds::ArrayNode{T}, e) where {T<:Mill.NGramMatrix}
	repr_boolean(:and, ds.data.s[participate(m)])
end

function repr(::MIME"text/json", m::EmptyMask, ds::ArrayNode{T}, e) where {T<:Mill.NGramMatrix}
	@info "Improve NGramMatrix with EmptyMask"
	repr_boolean(:and, ds.data.s)
end

#	We need at least one item from each cluster, which means that there is an OR relationship
# 	withing clusters and AND relationship across clusters 
#
#
function repr(mim::MIME"text/json", m::AbstractExplainMask, ds::BagNode, e)
    ismissing(ds.data) && Dict{Symbol, String}()

    #get indexes of contributing clusters
	contributing = participate(m) .& mask(m)
	cluster_membership = m.mask.cluster_membership
	contributing_clusters = unique(m.mask.cluster_membership[contributing])
	cluster2instance = idmap(m.mask.cluster_membership)

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
	repr_boolean(:and, ss)
end

function repr(mim::MIME"text/json", m::EmptyMask, ds::BagNode, e)
    ismissing(ds.data) && Dict{Symbol,String}()
    ss = repr(mim, m, ds.data, e.item);
    repr_boolean(:and, ss)
end

function repr(mim::MIME"text/json", m::AbstractExplainMask, ds::AbstractProductNode, e)
	s = map(sort(collect(keys(ds.data)))) do k
        subs = repr(mim, m[k], ds[k], e[k])
        isempty(subs) ? nothing : k => subs
    end
    Dict(filter(!isnothing, s))
end

