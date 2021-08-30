using ExplainMill
using ExplainMill: prune, Mask, invalidate!, foreach_mask, participate, mask, prunemask
using ExplainMill: MatrixMask, SparseArrayMask, CategoricalMask, ProductMask, BagMask
using ExplainMill: NGramMatrixMask, EmptyMask
using ExplainMill: FlatView, identifycolumns, create_mask_structure, ParticipationTracker
using Random, StatsBase
using Setfield
using Test
using Mill
using Mill: nobs
using Flux
using Duff
using SparseArrays
using HierarchicalUtils

include("specimen.jl")
# TODO test that explanations are indeed subsets of the original json
# TODO call e2boolean with full and sampled masks

@testset "ExplainMill.jl" begin
    include("structuralmasks.jl")
    include("flatmasks.jl")
    include("heuristics.jl")
    include("partialeval.jl")
    include("explain.jl")

    @testset "Logic output" begin 
    	@test_broken false
    end
end
