abstract type AbstractExplainMask end;
abstract type AbstractListMask <: AbstractExplainMask end;
abstract type AbstractNoMask <: AbstractExplainMask end;

RealArray = Union{Vector{T}, Matrix{T}} where {T<:Real}

NodeType(::Type{T}) where T <: AbstractListMask = LeafNode()
noderepr(n::T) where {T <: AbstractExplainMask} = "$(T.name)"

participate(m::AbstractExplainMask) = participate(m.mask)
Base.fill!(m::AbstractExplainMask, v) = Base.fill!(m.mask, v)
Base.fill!(m::AbstractNoMask, v) = nothing
index_in_parent(m, i) = i

function mapmask(f, m::AbstractListMask)
	(mask = f(m.mask),)
end


invalidate!(m::AbstractExplainMask) = invalidate!(m, Vector{Int}())

include("mask.jl")
include("densearray.jl")
include("sparsearray.jl")
include("categoricalarray.jl")
include("NGramMatrix.jl")
include("skip.jl")
include("bags.jl")
include("product.jl")
include("lazymask.jl")
include("parentstructure.jl")
include("flatmasks.jl")

"""
	Mask(ds, m, initstats, cluster)

	creates a mask containing `initstats` statistics for a given node 
	ds --- node 
	m  --- model for the node 
	initstats --- method costructing statistics for a given node using the output 
	of cluster function 
	cluster --- function (ds, m) -> returning vector identifying cluster membership for each explained item
"""
# function Mask(ds::AbstractNode, m::AbstractMillModel, initstats, cluster; verbose::Bool = false)
# 	@info "$(eltype(ds).node) is not supported"
# end

# function Mask(ds::AbstractNode, m::AbstractMillModel; verbose::Bool = false)
# 	Mask(ds, m, Daf, nocluster)
# end

mulmask(m::AbstractExplainMask) = mulmask(m.mask)
mulmask(m::Mask{Nothing,M}) where {M<:RealArray} = σ.(m.stats)
mulmask(m::Mask{Array{Int64,1},M}) where {M<:RealArray} = σ.(m.stats[m.cluster_membership,:])

@deprecate mask prunemask

prunemask(m::AbstractExplainMask) = prunemask(m.mask)
prunemask(m::Mask{Nothing,M}) where {M<:RealArray} = reshape(m.mask,:)
prunemask(m::Mask{Array{Int64,1},M}) where {M<:RealArray} = m.mask[m.cluster_membership,:][:]

prunemask(m::Mask{Nothing,M}) where {M<:Duff.Daf} = reshape(m.mask,:)
prunemask(m::Mask{Array{Int64,1},M}) where {M<:Duff.Daf} = m.mask[m.cluster_membership,:][:]

mulmask(m::Mask{Nothing,M}) where {M<:Duff.Daf} = prunemask(m)
mulmask(m::Mask{Array{Int64,1},M}) where {M<:Duff.Daf} = prunemask(m)

function updateparticipation!(ms)
	mapmask(m -> participate(m) .= true, ms)
	invalidate!(ms)
end
