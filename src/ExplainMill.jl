module ExplainMill
using Mill, Duff, SparseArrays, StatsBase, CatViews, Distances, Clustering
using Mill: paddedprint, COLORS
import Mill: dsprint
using TimerOutputs

const to = TimerOutput();

abstract type AbstractExplainMask end;
abstract type AbstractListMask <: AbstractExplainMask end;

participate(m::AbstractExplainMask) = participate(m.mask)
mask(m::AbstractExplainMask) = mask(m.mask)


function cluster_instances(x)
	d = pairwise(CosineDist(), x, dims = 2)
	dbscan(d, 0.2, 1).assignments
end


function mapmask(f, m::AbstractListMask)
	(mask = f(m.mask),)
end

invalidate!(m::AbstractExplainMask) = invalidate!(m, Vector{Int}())

Base.show(io::IO, ::MIME"text/plain", n::AbstractExplainMask) = dsprint(io, n)

include("mask.jl")
include("densearray.jl")
include("sparsearray.jl")
include("NGramMatrix.jl")
include("skip.jl")
include("bags.jl")
include("product.jl")
include("explain.jl")

Duff.update!(daf, mask::Nothing, v::Number, valid_columns = nothing) = nothing

export explain, dafstats

end # module
