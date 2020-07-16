using JsonGrinder

# TODO write using HUtils?

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

function print_explained(io, ds::ArrayNode{T}, e::E; pad = []) where {T<:Flux.OneHotMatrix, E<:ExtractDict}
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
        print_explained(io, n.data[ks[i]], e[String(ks[i])], pad=[pad; (c, "  │" * repeat(" ", max(3, 2+length(ks[i]))))])
    end
    println(io)
    paddedprint(io, "  └── $(ks[end]): ", color=c, pad=pad)
    print_explained(io, n.data[ks[end]], e[String(ks[end])], pad=[pad; (c, repeat(" ", 3+max(3, 2+length(ks[end]))))])
end

print_explained(ds::AbstractNode, e) = print_explained(stdout, ds, e)

print_explained(::MIME"text/plain",ds::AbstractNode, e) = print_explained(stdout, ds, e)

function print_explained(::MIME"text/html", ds::ArrayNode{T}, e::E; pad = []) where {T<:Flux.OneHotMatrix, E<:ExtractCategorical}
	c = COLORS[(length(pad)%length(COLORS))+1]
	x = ds.data
	idxs = map(argmax, x.data)
	filter!(i -> x.height != i, idxs)
	if isempty(idxs)
		return("")
	else
		idxs = unique(idxs);
		d = reversedict(e.keyvalemap);
		s = join(map(i -> d[i], idxs), " and ")
		return(s)
	end
end

function print_explained(m::MIME"text/html", ds::ArrayNode{T}, e; pad = []) where {T<:Mill.NGramMatrix}
	s = filter(!isempty, ds.data.s)
	isempty(s) && return("")
	p = mapfoldl(p -> p[2], *, pad)
	mapfoldl(s -> p*"<li>"*s*"</li>\n", *, s)
end

function print_explained(m::MIME"text/html", ds::BagNode, e; pad = [])
    c = COLORS[(length(pad)%length(COLORS))+1]
    if ismissing(ds.data)
	    return("")
	else
		s = print_explained(m, ds.data, e.item, pad = [pad; (c, "      ")])
		isempty(s) && return("")
	    return("List of\n"*s)
	end
end

function print_explained(m::MIME"text/html", ds::AbstractProductNode, e; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    # repr(m, "ProductNode", color=c)
    ks = sort(collect(keys(ds.data)))
    ss = map(k -> print_explained(m, ds.data[k], e[String(k)], pad=[pad; (c, "  " * repeat(" ", max(3, 2+length(k))))]), ks)
    mask = .!isempty.(ss)
    ks, ss = ks[mask], ss[mask]
    isempty(ks) && return("")
    "ProductNode\n"*mapfoldl(i -> String(ks[i])*": "*ss[i], *, 1:length(ks))
end
