abstract type AbstractExplainMask end;
abstract type AbstractListMask <: AbstractExplainMask end;
abstract type AbstractNoMask <: AbstractExplainMask end;

RealArray = Union{Vector{T}, Matrix{T}} where {T<:Real}

participate(m::AbstractExplainMask) = participate(m.mask)
Base.fill!(m::AbstractExplainMask, v) = Base.fill!(m.mask, v)
Base.fill!(m::AbstractNoMask, v) = nothing

function mapmask(f, m::AbstractListMask)
	(mask = f(m.mask),)
end

invalidate!(m::AbstractExplainMask) = invalidate!(m, Vector{Int}())

include("mask.jl")
include("parentstructure.jl")
include("flatview.jl")

include("nodemasks/densearray.jl")
include("nodemasks/sparsearray.jl")
include("nodemasks/categoricalarray.jl")
include("nodemasks/ngrammatrix.jl")
include("nodemasks/skip.jl")
include("nodemasks/bags.jl")
include("nodemasks/product.jl")
include("nodemasks/lazymask.jl")


"""
	prunemask(m)::Vector{Bool}

	vector of items that should be presented in the subset
"""
prunemask(m::AbstractExplainMask) = prunemask(m.mask)
prunemask(m::Mask{Nothing,M}) where {M} = prunemask(m.mask)
prunemask(m::Mask{Array{Int64,1},M}) where {M} = view(prunemask(m.mask), m.cluster_membership)

"""
	diffmask(m)

	differentiable version of an item that should be presented in the subset
"""
diffmask(m::AbstractExplainMask) = diffmask(m.mask)
diffmask(m::Mask{Nothing,M}) where {M<:RealArray} = diffmask(m.mask)
diffmask(m::Mask{Array{Int64,1},M}) where {M<:RealArray} = diffmask(m.mask)[m.cluster_membership]

function updateparticipation!(ms)
	mapmask(m -> participate(m) .= true, ms)
	invalidate!(ms)
end

Base.length(m::AbstractExplainMask) = length(m.mask)
Base.getindex(m::AbstractExplainMask, i) = m.mask[i]
Base.setindex!(m::AbstractExplainMask, v, i) = m.mask[i] = v