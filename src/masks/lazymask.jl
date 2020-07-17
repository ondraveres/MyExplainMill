Mask(::T, m::AbstractMillModel, initstats, cluster; verbose::Bool = false) where {T<:LazyNode} = ExplainMill.EmptyMask()

Mask(::T, initstats; verbose::Bool = false) where {T<:LazyNode} = ExplainMill.EmptyMask()

function Base.repr(::MIME{Symbol("text/json")}, ::ExplainMill.EmptyMask, ds::T, e) where {T<:LazyNode}
	repr_boolean(:and, ds.data)
end

function Base.repr(::MIME{Symbol("text/json")}, m::ExplainMill.Mask, ds::T, e) where {T<:LazyNode}
	repr_boolean(:and, ds.data[ExplainMill.participate(m) .& ExplainMill.prunemask(m)])
end

function Base.match(s::String, e, v::T; path = (), verbose = false) where {T<:LazyNode}
	printontrue(s ∈ v.data, verbose, path," ", s)
end

_nocluster(m::LazyModel{N}, ds::LazyNode{N}) where {N} = nobs(ds)

(m::LazyModel)(ds::LazyNode, ::ExplainMill.EmptyMask) = m(ds)