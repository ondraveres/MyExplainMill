abstract type AbstractExplainMask end;
abstract type AbstractListMask <: AbstractExplainMask end;
abstract type AbstractNoMask <: AbstractExplainMask end;

RealArray = Union{Vector{T}, Matrix{T}} where {T<:Real}

participate(m::AbstractExplainMask) = participate(m.mask)
Base.fill!(m::AbstractExplainMask, v) = Base.fill!(m.mask, v)
Base.fill!(m::AbstractNoMask, v) = nothing
index_in_parent(m, i) = i

function mapmask(f, m::AbstractListMask)
	(mask = f(m.mask),)
end


invalidate!(m::AbstractExplainMask) = invalidate!(m, Vector{Int}())

include("mask.jl")
include("parentstructure.jl")

include("densearray.jl")
include("sparsearray.jl")
include("categoricalarray.jl")
include("NGramMatrix.jl")
include("skip.jl")
include("bags.jl")
include("product.jl")
include("lazymask.jl")
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

@deprecate mask prunemask

"""
	prunemask(m)

	Mask on the level of observations / features / items used mainly in 
	`getindex`
"""
prunemask(m::AbstractExplainMask) = prunemask(m.mask)
prunemask(m::Mask{Nothing,M}) where {M} = reshape(m.mask,:)
# prunemask(m::Mask{Array{Int64,1},M}) where {M} = m.mask[m.cluster_membership,:][:]
prunemask(m::Mask{Array{Int64,1},M}) where {M} = view(m.mask, m.cluster_membership)

"""
	prunemask(m)

	Mask on the level of clusters used to turn on / off items in 
	explanations during explanation / ranking
"""
clustermask(m::AbstractExplainMask) = clustermask(m.mask)
clustermask(m::Mask) = reshape(m.mask,:)


"""
	mulmask(m)

	Mask used in GNN / Grad explainers with which observations / features / items are mutliplied
	after passed through `σ` function.
"""
mulmask(m::AbstractExplainMask) = mulmask(m.mask)
mulmask(m::Mask{Nothing,M}) where {M<:RealArray} = σ.(m.stats)
mulmask(m::Mask{Array{Int64,1},M}) where {M<:RealArray} = σ.(m.stats[m.cluster_membership,:])
mulmask(m::Mask{Nothing,M}) where {M<:Duff.Daf} = prunemask(m)
mulmask(m::Mask{Array{Int64,1},M}) where {M<:Duff.Daf} = prunemask(m)

function updateparticipation!(ms)
	mapmask(m -> participate(m) .= true, ms)
	invalidate!(ms)
end
