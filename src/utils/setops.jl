nleaves(s::T) where {T<:Union{Number,AbstractString}} = 1
nleaves(s::Nothing) = 0
nleaves(s::AbstractArray) = isempty(s) ? 0 : mapreduce(nleaves, +, s)
nleaves(d::Dict) = isempty(d) ? 0 : mapreduce(nleaves, +, values(d))

nnodes(s::T) where {T<:Union{Number,AbstractString}} = 1
nnodes(s::Nothing) = 0
nnodes(s::AbstractArray) = isempty(s) ? 0 : 1 + mapreduce(nnodes, +, s)
nnodes(d::Dict) = isempty(d) ? 0 : 1 + mapreduce(nnodes, +, values(d))

keys2symbol(d::Dict{String,A}) where {A} = Dict([Symbol(k) => v for (k,v) in d]...)
keys2symbol(d::Dict{Symbol,A}) where {A} = d

promote_keys(d1::Dict{String,A}, d2::Dict{String, B}) where {A,B} = (d1, d2)
promote_keys(d1::Dict{Symbol,A}, d2::Dict{Symbol, B}) where {A,B} = (d1, d2)
promote_keys(d1::Dict{String,A}, d2::Dict{Symbol, B}) where {A,B} = (keys2symbol(d1), d2)
promote_keys(d1::Dict{Symbol,A}, d2::Dict{String, B}) where {A,B} = (d1, keys2symbol(d2))

function jsondiff(d1::Dict, d2::Dict)
	d1, d2 = promote_keys(d1, d2)
	ps = map(collect(keys(d1))) do k 
		!haskey(d2, k) && return(k => d1[k])
		k => jsondiff(d1[k], d2[k])
	end
	ps = filter(p -> !isnothing(p.second), ps)
	ps = filter(p -> !isempty(p.second), ps)
	Dict(ps...)
end

function jsondiff(d1::AbstractString, d2::AbstractString)
	d1 == d2 ? nothing : d1
end

function jsondiff(d1::Number, d2::Number)
	d1 == d2 ? nothing : d1
end


function jsondiff(d1::Array{T}, d2::Array{U}) where {T<:Union{AbstractString, Number}, U<:Union{AbstractString, Number}}
	setdiff(d1, d2)
end

function jsondiff(d1::Array, d2::Array)
	@assert length(d1) == length(d2) == 1
	o = jsondiff(only(d1), only(d2))
	isnothing(o) && return(nothing)
	isempty(o) && return(nothing)
	o
end

isjsonsubtree(d1, d2) = d1 == d2
isjsonsubtree(d1::AbstractArray, d2::AbstractArray) = all(i1 -> any(i2 -> isjsonsubtree(i1,i2),d2), d1)
function isjsonsubtree(d1::Dict, d2::Dict) 
	d1, d2 = promote_keys(d1, d2)
	keys(d1) != keys(d2) && return(false)
	all(k -> isjsonsubtree(d1[k],d2[k]), keys(d1))
end

jsonmerge(d1, d2) = d2
jsonmerge(d1::AbstractArray, d2::AbstractArray) = vcat(d1,d2)
function jsonmerge(d1::Dict, d2::Dict) 
	d1, d2 = promote_keys(d1, d2)
	merge(d1, d2, Dict([k => jsonmerge(d1[k],d2[k]) for k in intersect(keys(d1),keys(d2))]))
end