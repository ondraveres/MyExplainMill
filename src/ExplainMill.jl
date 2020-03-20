module ExplainMill
using Mill, Duff, SparseArrays, StatsBase, CatViews
using Mill: paddedprint, COLORS
import Mill: dsprint

abstract type AbstractExplainMask end;
function Mask end;

invalidate!(mask::AbstractExplainMask) = invalidate!(mask, Vector{Int}())

Base.show(io::IO, ::MIME"text/plain", n::AbstractExplainMask) = dsprint(io, n)

include("densearray.jl")
include("sparsearray.jl")
include("NGramMatrix.jl")
include("bags.jl")
include("product.jl")

Duff.update!(daf, mask::Nothing, v::Number, valid_columns = nothing) = nothing

export explain, dafstats

include("hierarchical_utils.jl")

Base.show(io::IO, ::T) where T <: Union{AbstractExplainMask, TreeMask} = show(io, Base.typename(T))
Base.show(io::IO, ::MIME"text/plain", n::Union{AbstractExplainMask, TreeMask}) = HierarchicalUtils.printtree(io, n; trav=false)
Base.getindex(n::Union{JSONEntry, AbstractExtractor}, i::AbstractString) = HierarchicalUtils.walk(n, i)

end # module
