using ExplainMill
# TODO export these?
using ExplainMill: prune, Mask, invalidate!, mapmask, participate, mask, prunemask
using ExplainMill: MatrixMask, SparseArrayMask, CategoricalMask, ProductMask, BagMask
using ExplainMill: NGramMatrixMask, EmptyMask
using ExplainMill: FlatView, identifycolumns
using ExplainMill: sigmoid, scale201, fuseaffine!, minimax, rescale
using Random, StatsBase
using Test
using Mill
using Mill: nobs
using Flux
using Duff
using SparseArrays
using HierarchicalUtils

# TODO integration tests of the whole pipeline
# TODO test that explanations are indeed subsets of the original json
# TODO call e2boolean with full and sampled masks

@testset "ExplainMill.jl" begin
    include("structuralmasks.jl")
    include("flatmasks.jl")
    include("gnn.jl")
    include("explain.jl")
    include("sigmoid.jl")
end
