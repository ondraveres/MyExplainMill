using ExplainMill
using Mill
using Test
using Flux
using SparseArrays
using StatsBase: nobs
using ExplainMill: collectmasks

all_pruning_methods = [:Flat_HAdd, :Flat_HArr, :Flat_HArrft, :Flat_Gadd, :Flat_Garr, :Flat_Garrft, :LbyL_HAdd, :LbyL_HArr, :LbyL_HArrft, :LbyL_Gadd, :LbyL_Garr, :LbyL_Garrft]
@testset "Checking integration with pruner" begin
    ds = specimen_sample()
    model = reflectinmodel(ds, d -> Dense(d, 4), SegmentedMean)

    for e in [ConstExplainer(), StochasticExplainer(), GnnExplainer(200), GradExplainer(), ExplainMill.DafExplainer()]
        mk = stats(e, ds, model)

        o = softmax(model(ds).data)[:]
        τ = 0.9 * maximum(o) 
        class = argmax(softmax(model(ds).data)[:])
        f = () -> softmax(model(ds[mk]).data)[class] - τ
        @test !ExplainMill.flatsearch!(f, mk)
        @test !ExplainMill.levelbylevelsearch!(f, mk)

        @test !ExplainMill.flatsearch!(f, mk, random_removal = true)
        @test !ExplainMill.levelbylevelsearch!(f, mkrandom_removal = true)

        # this test is more like a test that it passes
        for pruning_method in all_pruning_methods
            @test explain(e, ds, model, rel_tol=0.9, pruning_method) isa ExplainMill.AbstractStructureMask
        end
    end
end