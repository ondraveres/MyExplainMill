module ExplainMill
using Mill
using Duff
using SparseArrays
using StatsBase
using Distances
using Clustering
using Flux
using Zygote
using JSON
using HierarchicalUtils
using DataFrames
using JSON
using JsonGrinder
using Setfield
using DataStructures


output(ds::ArrayNode) = ds.data
output(x::AbstractArray) = x
NNlib.softmax(ds::ArrayNode) = ArrayNode(softmax(ds.data))

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
export prunemask, diffmask, HeuristicMask, SimpleMask
include("output/logic_output.jl")

include("heuristics/const_explainer.jl")
include("heuristics/daf_explainer.jl")
include("heuristics/gnn_explainer.jl")
include("heuristics/grad_explainer.jl")
include("heuristics/stochastic_explainer.jl")
export GnnExplainer, GradExplainer, ConstExplainer, StochasticExplainer
export stats, heuristic

include("prunemissing.jl")
include("sigmoid.jl")
include("predict.jl")
include("sampler.jl")
include("stats.jl")
include("matching.jl")
include("ensemble.jl")
include("pruning/pruning.jl")
include("utils/entropy.jl")
include("utils/setops.jl")
include("utils/partialeval.jl")
include("utils/lensutils.jl")
include("explain.jl")
include("distances/fisher.jl")
include("distances/clusterings.jl")
include("distances/manual.jl")

include("hierarchical_utils.jl")

Base.show(io::IO, ::T) where T <: AbstractStructureMask = print(io, Base.nameof(T))
Base.show(io::IO, ::MIME"text/plain", n::AbstractStructureMask) = HierarchicalUtils.printtree(io, n; trav=false, htrunc=3, vtrunc=20)
Base.getindex(n::AbstractStructureMask, i::AbstractString) = HierarchicalUtils.walk(n, i)

export explain, e2boolean, predict, confidence, prunemissing, prune, e2boolean
export removeabsent, removemissing
end # module
