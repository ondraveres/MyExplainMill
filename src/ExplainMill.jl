module ExplainMill
using Mill, Duff, SparseArrays, StatsBase, CatViews, Distances, Clustering, Flux
using HierarchicalUtils, JsonGrinder
import HierarchicalUtils: NodeType, childrenfields, children, InnerNode, SingletonNode, LeafNode, printtree, noderepr
using TimerOutputs

const to = TimerOutput();

function dbscan_cosine(x, ϵ)
	nobs(x) == 1 && return([1])
	d = pairwise(CosineDist(), x, dims = 2)
	dbscan(d, ϵ, 1).assignments
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

Duff.update!(daf, mask::Nothing, v::Number, valid_columns = nothing) = nothing

include("masks/masks.jl")
include("output/logic_output.jl")
include("output/prettyprint.jl")
include("dafstats.jl")
include("explain.jl")
include("prunemissing.jl")
include("sigmoid.jl")
include("predict.jl")
include("sampler.jl")
include("stats.jl")
include("matching.jl")
include("gnn_explainer.jl")


export explain, print_explained, e2boolean, predict, confidence, prunemissing, prune, e2boolean

Base.show(io::IO, ::T) where T <: AbstractExplainMask = show(io, Base.typename(T))
Base.show(io::IO, ::MIME"text/plain", n::AbstractExplainMask) = HierarchicalUtils.printtree(io, n; trav=false)
Base.getindex(n::AbstractExplainMask, i::AbstractString) = HierarchicalUtils.walk(n, i)

end # module
