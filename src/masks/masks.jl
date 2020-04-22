abstract type AbstractExplainMask end;
abstract type AbstractListMask <: AbstractExplainMask end;

NodeType(::Type{T}) where T <: AbstractListMask = LeafNode()
noderepr(n::AbstractExplainMask) = "$(Base.typename(typeof(n)))"

participate(m::AbstractExplainMask) = participate(m.mask)
mask(m::AbstractExplainMask) = mask(m.mask)

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
