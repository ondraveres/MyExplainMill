module ExplainMill
using Mill, Duff, SparseArrays, StatsBase, CatViews, Distances, Clustering, Flux
using HierarchicalUtils, JsonGrinder
import HierarchicalUtils: NodeType, childrenfields, children, InnerNode, SingletonNode, LeafNode, printtree, noderepr
using TimerOutputs

const to = TimerOutput();

abstract type AbstractExplainMask end;
abstract type AbstractListMask <: AbstractExplainMask end;

NodeType(::Type{T}) where T <: AbstractListMask = LeafNode()
noderepr(n::AbstractExplainMask) = "$(Base.typename(typeof(n)))"

participate(m::AbstractExplainMask) = participate(m.mask)
mask(m::AbstractExplainMask) = mask(m.mask)

function cluster_instances(x)
	nobs(x) == 1 && return([1])
	d = pairwise(CosineDist(), x, dims = 2)
	dbscan(d, 0.2, 1).assignments
end

function idmap(ids::Vector{T}) where{T}
	d = Dict{T,Vector{Int}}()
	for (i,v) in enumerate(ids)
		if haskey(d, v)
			d[v] = vcat(d[v], [i])
		else
			d[v] = [i]
		end
	end
	return(d)
end


function mapmask(f, m::AbstractListMask)
	(mask = f(m.mask),)
end

invalidate!(m::AbstractExplainMask) = invalidate!(m, Vector{Int}())

include("masks/mask.jl")
include("masks/densearray.jl")
include("masks/sparsearray.jl")
include("masks/categoricalarray.jl")
include("masks/NGramMatrix.jl")
include("masks/skip.jl")
include("masks/bags.jl")
include("masks/product.jl")
include("output/logic_output.jl")
include("output/prettyprint.jl")
include("explain.jl")
include("removemissing.jl")
include("sigmoid.jl")
include("predict.jl")

Duff.update!(daf, mask::Nothing, v::Number, valid_columns = nothing) = nothing

export explain, dafstats, print_explained

Base.show(io::IO, ::T) where T <: AbstractExplainMask = show(io, Base.typename(T))
Base.show(io::IO, ::MIME"text/plain", n::AbstractExplainMask) = HierarchicalUtils.printtree(io, n; trav=false)
Base.getindex(n::AbstractExplainMask, i::AbstractString) = HierarchicalUtils.walk(n, i)

end # module
