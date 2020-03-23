module ExplainMill
using Mill, Duff, SparseArrays, StatsBase, CatViews, Distances, Clustering, Flux
# using Mill: paddedprint, COLORS
# import Mill: dsprint
using TimerOutputs

const to = TimerOutput();

abstract type AbstractExplainMask end;
abstract type AbstractListMask <: AbstractExplainMask end;

participate(m::AbstractExplainMask) = participate(m.mask)
mask(m::AbstractExplainMask) = mask(m.mask)

function cluster_instances(x)
	nobs(x) == 1 && return([1])
	d = pairwise(CosineDist(), x, dims = 2)
	dbscan(d, 0.2, 1).assignments
end


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
include("explain.jl")
include("removemissing.jl")
include("prettyprint.jl")
include("sigmoid.jl")

Duff.update!(daf, mask::Nothing, v::Number, valid_columns = nothing) = nothing

export explain, dafstats, print_explained

include("hierarchical_utils.jl")

Base.show(io::IO, ::T) where T <: AbstractExplainMask = show(io, Base.typename(T))
Base.show(io::IO, ::MIME"text/plain", n::AbstractExplainMask) = HierarchicalUtils.printtree(io, n; trav=false)
Base.getindex(n::AbstractExplainMask, i::AbstractString) = HierarchicalUtils.walk(n, i)

end # module
