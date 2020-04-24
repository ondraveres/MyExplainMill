abstract type AbstractExplainMask end;
abstract type AbstractListMask <: AbstractExplainMask end;
abstract type AbstractNoMask <: AbstractExplainMask end;

NodeType(::Type{T}) where T <: AbstractListMask = LeafNode()
noderepr(n::T) where {T <: AbstractExplainMask} = "$(T.name)"

participate(m::AbstractExplainMask) = participate(m.mask)
mask(m::AbstractExplainMask) = mask(m.mask)
Base.fill!(m::AbstractExplainMask, v) = Base.fill!(mask(m), v)
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
function Mask(ds::AbstractNode, m::AbstractMillModel, initstats, cluster; verbose::Bool = false)
	@info "$(eltype(ds).node) is not supported"
end

function Mask(ds::AbstractNode, m::AbstractMillModel; verbose::Bool = false)
	Mask(ds, m, Daf, nocluster)
end

gnnmask(m::AbstractExplainMask) = σ.(m.mask.stats)