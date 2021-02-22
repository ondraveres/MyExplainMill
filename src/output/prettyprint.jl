using JsonGrinder

reversedict(d) = Dict([v => k for (k,v) in d]...)

# copied from Mill.jl, which now does not have it
const COLORS = [:blue, :red, :green, :yellow, :cyan, :magenta]

function infer_sample_child(e::E, n::AbstractProductNode, name::S) where {E<:ExtractDict, S<:Symbol}
	e_vec_keys = keys(e.vec)
	if name ∈ e_vec_keys
		scalars_idx = findfirst(isequal(ks[i]), e_vec_keys |> collect)
		# I'm throwing off metadata, but that's because I think they are not used at this point
		return ArrayNode(n[:scalars].data[scalars_idx:scalars_idx, :])
	else
		return n[name]
	end
end

function paddedprint(io, s...; color=:default, pad=[])
    for (c, p) in pad
        printstyled(io, p, color=c)
    end
    printstyled(io, s..., color=color)
end


function paddedprint(io::IO, d::Dict{String, T}; color=:default, pad=[]) where {T}
	c = COLORS[(length(pad)%length(COLORS))+1]
	isempty(d) && paddedprint(io, "∅\n", color = c, pad = pad)
	m = maximum(length(k) for k in keys(d))
    for (i, (k,v)) in enumerate(d)
    	s = rpad(k, m)*" => $(v)"
    	s *= i == length(d) ? "" : "\n"
    	paddedprint(io, s, color = c, pad = pad)
    end
end

function print_explained(io, ds::Missing, e; pad = [])
	c = COLORS[(length(pad)%length(COLORS))+1]
	paddedprint(io, "∅", color = c)
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
		idxs = unique(idxs)
		d = reversedict(e.keyvalemap)
		s = join(map(i -> d[i], idxs), " and ")
		paddedprint(io, s, color = c)
	end
end

function extract_scalar_inv(e::E, vals::T) where {E<:ExtractScalar, T<:Vector}
	# basically inverting the formula in
	# https://github.com/pevnak/JsonGrinder.jl/blob/master/src/extractors/extractscalar.jl#L34
	if e.datatype <: Integer
		return convert.(e.datatype, round.(e.datatype, vals ./ e.s) .+ e.c)
	else
		convert.(e.datatype, vals ./ e.s .+ e.c)
	end
end

# shortcuit for skipping child in case of single child of dict
function print_explained(io, ds::BagNode, e::ExtractDict{S,V}; pad = []) where {S<:Nothing,V<:Dict}
	nchildren(e) > 1 && error("This really should not be happening")
	print_explained(io, ds, e.other |> values |> first, pad = pad)
end

function print_explained(io::IO, n::AbstractProductNode, e::ExtractDict{S,V}; pad=[]) where {S<:Dict,V<:Union{Dict,Nothing}}
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io, "ProductNode", color=c)
    ks = sort(collect(keys(n.data)))
	# must do the hack, there's no other way
	scalar_names = keys(e.vec)
	if :scalars ∈ ks
		ks = [filter(k->k!=:scalars, ks)..., scalar_names...]
	end
	m = length(ks)
    for i in 1:(m-1)
        println(io)
		s_child = infer_sample_child(e, n, ks[i])
        paddedprint(io, "  ├── $(ks[i]): ", color=c, pad=pad)
        print_explained(io, s_child, e[ks[i]], pad=[pad; (c, "  │" * repeat(" ", max(3, 2+length(String(ks[i])))))])
    end

    println(io)
	s_child = infer_sample_child(e, n, ks[end])
    paddedprint(io, "  └── $(ks[i]): ", color=c, pad=pad)
    print_explained(io, s_child, e[ks[end]], pad=[pad; (c, repeat(" ", 3+max(3, 2+length(String(ks[end])))))])
end

# when S<:Nothing we don't have to care about :scalars
function print_explained(io::IO, n::AbstractProductNode, e::ExtractDict{S,V}; pad=[]) where {S<:Nothing,V<:Dict}
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
    paddedprint(io, "  └── $(ks[i]): ", color=c, pad=pad)
    print_explained(io, n[ks[end]], e[ks[end]], pad=[pad; (c, repeat(" ", 3+max(3, 2+length(String(ks[end])))))])
end

function print_explained(io, ds::ArrayNode{T}, e::E; pad = []) where {T<:Matrix, E<:ExtractScalar}
	c = COLORS[(length(pad)%length(COLORS))+1]
	x = ds.data
	idxs = map(argmax, x)
	# based on https://avast-software.slack.com/archives/C016UGWTP6K/p1597408538013500 I'm doing a nasty quickfix here
	if isempty(idxs) || all(x .== 0)
		s = "∅"
		paddedprint(io, s, color = c)
	else
		# todo: test if it's behaving correctly, because it looks like it's producing 0 instead of missing data
		idxs = unique(idxs)
		s = join(extract_scalar_inv(e, x[idxs]), " and ")
		paddedprint(io, s, color = c)
	end
end

function print_explained(io, ds::ArrayNode{T}, e::E; pad = []) where {T<:Flux.OneHotMatrix, E<:ExtractDict}
	length(e.other) > 1 && @error "This should not happen"
	k = collect(keys(e.other))[1]
	print_explained(io, ds, e.other[k])
end

function print_explained(io, ds::ArrayNode{T}, e::E; pad = []) where {T<:Matrix, E<:ExtractDict}
	!isnothing(e.other) && @error "This should not happen"
	c = COLORS[(length(pad)%length(COLORS))+1]
	paddedprint(io, "matrix should be printed", color = c)
end

function print_explained(io, ds::ArrayNode{T}, e; pad = []) where {T<:Mill.NGramMatrix}
    c = COLORS[(length(pad)%length(COLORS))+1]
	s = filter(!isempty, ds.data.s)
	if isempty(s)
		s = "∅"
		paddedprint(io, s, color = c)
	else
		paddedprint(io,"\n", color=c)
		paddedprint(io, countmap(s), color = c, pad = pad)
	end
end

function print_explained(io, ds::BagNode, e; pad = [])
    c = COLORS[(length(pad)%length(COLORS))+1]
    if ismissing(ds.data) || isnothing(ds.data)
	    paddedprint(io, "∅", color = c)
	else
	    paddedprint(io, "List of\n", color=c)
	    paddedprint(io, "    └── ", color=c, pad=pad)
	    print_explained(io, ds.data, e.item, pad = [pad; (c, "      ")])
	end
end

function print_explained(io::IO, ds::T, e; pad = []) where {T<:Mill.LazyNode}
    c = COLORS[(length(pad)%length(COLORS))+1]
    if ismissing(ds.data) || isnothing(ds.data) || isempty(ds.data)
	    paddedprint(io, "∅", color = c)
	else
	    paddedprint(io, "$(T.name)\n", color=c)
	    paddedprint(io, countmap(ds.data), color = c, pad = pad)
	end
end

print_explained(ds::AbstractNode, e) = print_explained(stdout, ds, e)
print_explained(ds::Missing, e) = print_explained(stdout, ds, e)