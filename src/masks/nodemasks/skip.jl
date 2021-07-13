struct EmptyMask <: AbstractNoMask
end

function Base.getproperty(F::EmptyMask, d::Symbol)
	d == :child && return(EmptyMask())
	error("EmptyMask does not have a property $(d)")
end

Base.getindex(m::EmptyMask,i...) = EmptyMask()

function StatsBase.sample!(pruning_mask::EmptyMask)
end

mask(::EmptyMask) = Vector{Bool}()

participate(::EmptyMask) = Vector{Bool}()

prune(ds, mk::EmptyMask) = ds

function Base.getindex(ds, mk::EmptyMask, presentobs = fill(true, nobs(ds))) 
	all(presentobs) && return(ds)
	ds[presentobs]
end

foreach_mask(f, mk::EmptyMask, level = 1) = nothing
mapmask(f, mk::EmptyMask, level = 1) = mk

invalidate!(mk::EmptyMask, observations::Vector{Int}) = nothing

function (m::Mill.ArrayModel)(ds::ArrayNode, mk::EmptyMask)
    m(ds)
end
