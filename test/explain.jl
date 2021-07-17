using ExplainMill
using Mill
using Test
using Flux
using SparseArrays
using StatsBase: nobs
using ExplainMill: collectmasks

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
    end
end