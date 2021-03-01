using ExplainMill
# TODO export these?
using ExplainMill: prune, Mask, invalidate!, mapmask, participate, mask, prunemask
using ExplainMill: MatrixMask, SparseArrayMask, CategoricalMask, ProductMask, BagMask
using ExplainMill: NGramMatrixMask, EmptyMask
using ExplainMill: FlatView, index_in_parent, identifycolumns
using ExplainMill: sigmoid, scale201, fuseaffine!, minimax, rescale
using Random, StatsBase
using Test
using Mill
using Mill: nobs
using Flux
using Duff
using SparseArrays
using HierarchicalUtils

@testset "ExplainMill.jl" begin
    include("maskstructure.jl")
    include("flatmasks.jl")
    include("gnn.jl")
    include("explain.jl")
    include("sigmoid.jl")
end