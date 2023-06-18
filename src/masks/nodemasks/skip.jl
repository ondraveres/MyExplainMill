struct EmptyMask <: AbstractNoMask
end

function Base.getproperty(F::EmptyMask, d::Symbol)
	d == :child && return(EmptyMask())
	error("EmptyMask does not have a property $(d)")
end

Base.getindex(::EmptyMask, i...) = EmptyMask()

function StatsBase.sample!(pruning_mask::EmptyMask)
end

mask(::EmptyMask) = Vector{Bool}()

participate(::EmptyMask) = Vector{Bool}()

present(::EmptyMask, obs) = obs

prune(ds, ::EmptyMask) = ds

function Base.getindex(ds::AbstractMillNode, mk::EmptyMask, presentobs = fill(true, numobs(ds))) 
	all(presentobs) && return(ds)
	ds[presentobs]
end

foreach_mask(f, mk::EmptyMask, level, visited) = nothing
mapmask(f, mk::EmptyMask, level, visited) = mk

invalidate!(mk::EmptyMask, observations::Vector{Int}) = nothing

function (m::Mill.ArrayModel)(ds::ArrayNode, mk::EmptyMask)
    m(ds)
end
