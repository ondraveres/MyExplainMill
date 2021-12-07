import JSON: lower
import Base: ==, hash

"""
    Absent
	represent a part of an explanation which is not important
"""
struct Absent end

"""
    absent
	The singleton instance of type [`Absent`](@ref) representing an absent part of explanation
"""
const absent = Absent()

"""
    isabsent(x)
Indicate whether `x` is [`absent`](@ref).
"""
isabsent(::Any) = false
isabsent(::Absent) = true
Base.isempty(::Absent) = true


zeroobs() = absent
emptyexportobs() = Vector{Absent}()

using FillArrays
function OR(xs)
	# xs = removeabsent(unique(xs))
	xs = unique(xs)
	isempty(xs) && return(absent)
	length(xs) > 1 ? LogicalOR(xs) : only(xs)
end
OR(xs::Absent) = absent
struct LogicalOR{T}
	or::T
end
e1::LogicalOR == e2::LogicalOR = e1.or == e2.or
hash(e::LogicalOR, h::UInt) = hash(e.or, h)
Base.show(io::IO, mime::MIME"text/plain", a::LogicalOR) = println(io, "OR: ", a.or)

Base.isempty(x::LogicalOR) = isempty(x.or)
# few features to print Absent and serialize it to json
Base.show(io::IO, ::ExplainMill.Absent) = print(io, "absent")
JSON.lower(::Absent) = nothing

# OR(xs) = length(xs) > 1 ? OR(filter(!isabsent, xs)) : only(xs)

function dictofindexes(targets)
    d = Dict{Int,Vector{Int}}()
    for (i, v) in enumerate(targets)
        if haskey(d, v)
            push!(d[v], i)
        else
            d[v] = [i]
        end
    end
    return(d)
end

"""
    addor(m::Mask, x)

returns "or" relationships if they are needed in the explanation, by
substituing each item with an "or" of items in clusters

`m` is the mask and
`x` is the output of the explanation of bottom layers
"""
function addor(m::Mask{I, D}, x, active) where {I<:Vector{Int}, D}
    xi = view(m.cluster_membership, active)
    d = dictofindexes(xi)
    groups = Dict(map(k -> k => OR(x[d[k]]), collect(keys(d))))
    map(1:length(x)) do i
        isabsent(x[i]) ? absent : groups[xi[i]]
    end
end

addor(m::AbstractVectorMask, x, active) = x

addor(m::Mask{<:Vector{Int}}, x::Absent, active) = absent
addor(m::Mask{<:Nothing}, x, active) = x
addor(m::Mask{<:Nothing}, x::Absent, active) = absent
addor(m::AbstractStructureMask, x, active) = addor(m.mask, x, active)
addor(m::EmptyMask, x, active) = x

_retrieve_obs(::LazyNode, i) = error("LazyNode in Mill.jl does not support metadata (yet)")

_retrieve_obs(ds::ArrayNode{<:NGramMatrix, Nothing}, i) = ds.data.s[i]
_retrieve_obs(ds::ArrayNode{<:Flux.OneHotMatrix, Nothing}, i) = ds.data.data[i].ix
_retrieve_obs(ds::ArrayNode{<:Matrix, Nothing}, i, j) = ds.data[i, j]
_retrieve_obs(ds::ArrayNode{<:AbstractMatrix, <:AbstractMatrix}, i, j) = ds.metadata[i, j]
_retrieve_obs(ds::ArrayNode{<:AbstractMatrix, <:AbstractVector}, j) = ds.metadata[j]

_reversedict(d) = Dict([v => k for (k,v) in d]...)

"""
	contributing(m)

	returns a mask of items contributing to the explanation
"""
contributing(m::AbstractStructureMask, _) = prunemask(m.mask)
contributing(m::EmptyMask, l) = Fill(true, l)

function yarason(ds::ArrayNode{<:Flux.OneHotMatrix, M}, m::AbstractStructureMask, e::ExtractCategorical, exportobs=fill(true, nobs(ds))) where M
    c = contributing(m, nobs(ds))
    x = map(i -> c[i] ? _retrieve_obs(ds, i) : absent, findall(exportobs))
    if M === Nothing
        d = _reversedict(e.keyvalemap)
        x = map(i -> isabsent(i) ? i : get(d, i, "__UNKNOWN__"), x)
    end
    addor(m, x, exportobs)
end

"""
	unscale(x,e)

	original values of `x` before the extraction by `e` in json
"""
unscale(x::AbstractArray, e::ExtractScalar) = map(x -> unscale(x,e), x)
unscale(x::Number, e::ExtractScalar) = x / e.s + e.c
unscale(x::Absent, e::ExtractScalar) = absent
unscale(x, e) = x

"""
    yarason(ds, m, e, exportobs::Vector{Bool})

    Values for items in `ds` corresponding to `true` in `prunemask(m)`
    and `participating(m)`, or `absent` otherwise. Values are exported
    only for those indicated in binary mask `exportobs`. The export
    also takes into the account "clusters", which are exported using the
    `OR` as `OR
"""
function yarason(ds::ArrayNode{<:Matrix, M}, m, e::ExtractScalar, exportobs=fill(true, nobs(ds))) where M
    rows, cols = size(ds.data)
    c = contributing(m, size(ds.data,1))
    x = map(findall(exportobs)) do j
        [c[i] ? _retrieve_obs(ds, i, j) : absent for i in 1:rows]
    end
    M === Nothing ? unscale(x, e) : x
end

function yarason(ds::ArrayNode{<:Matrix, M}, m::ObservationMask, e::ExtractScalar, exportobs=fill(true, nobs(ds))) where M
    rows, cols = size(ds.data)
    c = contributing(m, cols)
    x = map(findall(exportobs)) do j
        [c[j] ? _retrieve_obs(ds, i, j) : absent for i in 1:rows]
    end
    M === Nothing ? unscale(x, e) : x
end

function yarason(ds::ArrayNode{<:Matrix, <:AbstractVector{M}}, m, e::ExtractScalar, exportobs=fill(true, nobs(ds))) where M <: AbstractVector
    @error "yarason for ArrayNodes carrying FeatureVectors (fixed size vectors) not supported yet."
end

function yarason(ds::ArrayNode{<:NGramMatrix}, m, e::ExtractString, exportobs = fill(true, nobs(ds)))
    c = contributing(m, nobs(ds))
    x = map(i -> c[i] ? _retrieve_obs(ds, i) : absent, findall(exportobs))
    addor(m, x, exportobs)
end

function yarason(ds::LazyNode, m, e, exportobs=fill(true, nobs(ds)))
    c = contributing(m, nobs(ds))
    x = map(i -> c[i] ? _retrieve_obs(ds, i) : absent, findall(exportobs))
    addor(m, x, exportobs)
end

function yarason(ds::BagNode, m, e::T, exportobs=fill(true, nobs(ds))) where T <: Union{JsonGrinder.ExtractKeyAsField, JsonGrinder.ExtractArray}
    nobs(ds) == 0 && return(zeroobs())
    isabsent(ds.data) && return(fill(absent, sum(exportobs)))
    nobs(ds.data) == 0 && return(fill(absent, sum(exportobs)))
    !any(exportobs) && emptyexportobs()

    #get indexes of c clusters
	present_childs = Vector(contributing(m, nobs(ds.data)))
	for b in ds.bags[.!exportobs]
	    present_childs[b] .= false
	end

	x = yarason(ds.data, m.child, _echild(e), present_childs)
	x = addor(m, x, present_childs)
	bags = Mill.adjustbags(ds.bags, present_childs)[exportobs]
	map(b -> unique(x[b]), bags)
end

_echild(e::JsonGrinder.ExtractArray) = e.item
_echild(e::JsonGrinder.ExtractKeyAsField) = e

function yarason(ds::ProductNode{T,M}, m, e::JsonGrinder.ExtractDict{<:Dict,<:Dict}, exportobs=fill(true, nobs(ds))) where {T<:NamedTuple, M}
    nobs(ds) == 0 && return(zeroobs())
    !any(exportobs) && return(emptyexportobs())

    o = _exportother(ds, m, e, exportobs)

    if :scalars ∈ setdiff(keys(ds), keys(e.other))
        o = map(d -> merge(d...), zip(o, _exportmatrix(ds[:scalars], m[:scalars], e.vec, exportobs)))
    end
    o
end

function yarason(ds::ProductNode{T,M}, m, e::JsonGrinder.ExtractDict{Nothing,<:Dict}, exportobs=fill(true, nobs(ds))) where {T<:NamedTuple, M}
    nobs(ds) == 0 && return(zeroobs())
    !any(exportobs) && return(emptyexportobs())
    _exportother(ds, m, e, exportobs)
end

function yarason(ds::ProductNode{T,M}, m, e::JsonGrinder.ExtractDict{<:Dict,Nothing}, exportobs=fill(true, nobs(ds))) where {S<: Dict, T<:NamedTuple, M}
    nobs(ds) == 0 && return(zeroobs())
    !any(exportobs) && return(emptyexportobs())
    _exportmatrix(ds[:scalars], m[:scalars], e.vec, exportobs)
end

function _exportother(ds::ProductNode, m, e::JsonGrinder.ExtractDict, exportobs)
	s = map(sort(collect(intersect(keys(ds.data), keys(e.other))))) do k
        k => yarason(ds[k], m[k], e.other[k], exportobs)
    end
    _arrayofdicts(Dict(s), sum(exportobs))
end

function _arrayofdicts(d::Dict, l)
    isempty(d) && return(fill(d, l))
    ks = collect(keys(d))
    map(1:l) do i
        Dict(map(k -> k => isabsent(d[k]) ? absent : d[k][i], ks))
    end
end

function _exportmatrix(args...)
    @error "Cannot extract metadata from an ArrayNode, did you extract with `store_input=true`?"
end
function _exportmatrix(ds::ArrayNode{T, <:AbstractMatrix},  m::ExplainMill.MatrixMask, e::Dict, exportobs=fill(true, nobs(ds))) where T
    x = yarason(ds, m, ExtractScalar(Float32, 0, 1), exportobs)
    map(x -> _parcel(x, e), x)
end

function _exportmatrix(ds::ArrayNode{T, <:AbstractMatrix},  m::ExplainMill.ObservationMask, e::Dict, exportobs=fill(true, nobs(ds))) where T
    x = yarason(ds, m, ExtractScalar(Float32, 0, 1), exportobs)
    map(x -> _parcel(x, e), x)
end

function _parcel(x::Vector{T}, e) where {T}
    d = Dict{Symbol,T}()
    for (offset, (k, f)) in enumerate(e)
        d[k] = unscale(x[offset], f)
    end
    d
end

# x = vcat([f(get(v,String(k),nothing)) for (k,f) in s.vec]...)

function yarason(ds::ProductNode, m, e::JsonGrinder.MultipleRepresentation, exportobs=fill(true, nobs(ds)))
	nobs(ds) == 0 && return(zeroobs())
	!any(exportobs) && emptyexportobs()
	s = map(sort(collect(keys(ds.data)))) do k
        yarason(ds[k], m[k], e.extractors[k], exportobs)
    end
    reduce(mergeexplanations, s)
end

mergeexplanations(a,b) = map(x -> logicaland(x...), zip(a,b))
logicaland(a::Vector, b::Vector) = intersect(a,b)
logicaland(::Absent, a) = a
logicaland(a, ::Absent) = a
logicaland(a, b) = a
logicaland(::Absent, ::Absent) = absent
logicaland(a::T, ::LogicalOR) where {T<:Union{String, Number, Vector}} = a
logicaland(a::LogicalOR, ::T) where {T<:Union{String, Number, Vector}} = a
logicaland(a::LogicalOR, b::LogicalOR) = OR(intersect(a.or, b.or))
function logicaland(a::String, b::String)
	a == "__UNKNOWN__" && return(b)
	a
end

function yarason(ds::ProductNode{T,M}, m, e::JsonGrinder.ExtractKeyAsField, exportobs=fill(true, nobs(ds))) where {T<:NamedTuple, M}
	k = yarason(ds[:key], m[:key],  e.key, exportobs)
	d = yarason(ds[:item], m[:item],  e.item, exportobs)
	map(x -> Dict(x[1] => x[2]), zip(k, d))
end

removeabsent(x) = x
removeabsent(x::Nothing) = missing

function removeabsent(x::Vector)
    x = map(removeabsent, x);
    x = filter(!isabsent, x);
    isempty(x) && return(absent)
    T = mapreduce(typeof, promote_type, x)
    x = Vector{T}(x)
    x
end

removeabsent(x::Vector{<:Absent}) = absent
removeabsent(x::LogicalOR) = LogicalOR(removeabsent(x.or))

# we need to keep absents, because if there will be key as value and it will depend on key, it will
# be pruned out
function removeabsent(d::Dict)
    x = map(k -> k => removeabsent(d[k]), collect(keys(d)))
    x = filter(a -> !isabsent(a.second), x)
    x = filter(a -> ismissing(a.second) || a.second == "" || !isempty(a.second), x)
    isempty(x) ? absent : Dict(x)
end

function e2boolean(dss::AbstractNode, pruning_mask, extractor)
    removeabsent(yarason(dss, pruning_mask, extractor))
end
