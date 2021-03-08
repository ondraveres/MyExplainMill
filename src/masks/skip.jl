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

prune(ds, mask::EmptyMask) = ds

function Base.getindex(ds, mask::EmptyMask, presentobs = fill(true, nobs(ds))) 
	all(presentobs) && return(ds)
	ds[presentobs]
end

mapmask(f, mask::EmptyMask) = nothing

invalidate!(mask::EmptyMask, observations::Vector{Int}) = nothing

function (m::Mill.ArrayModel)(ds::ArrayNode, mask::EmptyMask)
    m(ds)
end

index_in_parent(m::CategoricalMask, i) = error("Does not make sense")
