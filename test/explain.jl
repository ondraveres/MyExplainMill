using ExplainMill
using Mill
using Test
using Flux
using SparseArrays
using StatsBase: nobs
using ExplainMill: collectmasks

using ExplainMill: DafMask, create_mask_structure, ParticipationTracker, updateparticipation!

@testset "Check that heuristic values are non-zeros" begin
    ds = specimen_sample()
    model = reflectinmodel(ds, d -> Dense(d, 4), SegmentedMean)

    for e in [ConstExplainer(), StochasticExplainer(), GnnExplainer(200), GradExplainer(), ExplainMill.DafExplainer()]
        t = @elapsed mk = stats(e, ds, model)
        # println(typeof(e), " ", t)    # this is for debugging purposes
        @test all(sum(abs.(heuristic(m))) > 0 for m in collectmasks(mk))
    end
end

@testset "Checking integration with pruner" begin
    ds = specimen_sample()
    model = reflectinmodel(ds, d -> Dense(d, 4), SegmentedMean)

    for e in [ConstExplainer(), StochasticExplainer(), GnnExplainer(200), GradExplainer(), ExplainMill.DafExplainer()]
        mk = stats(e, ds, model)

        o = softmax(model(ds).data)[:]
        τ = 0.9 * maximum(o) 
        class = argmax(softmax(model(ds).data)[:])
        f = () -> softmax(model(ds[mk]).data)[class] - τ
        ExplainMill.flatsearch!(f, mk, random_removal = false, fine_tuning = false)
        ExplainMill.levelbylevelsearch!(mk, model, ds, τ, class)
    end
end