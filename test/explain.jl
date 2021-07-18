using ExplainMill
using Mill
using Test
using Flux
using SparseArrays
using StatsBase: nobs
using ExplainMill: logitconfgap

# heuristic_pruning_methods = [:Flat_HAdd, :Flat_HArr, :Flat_HArrft, :LbyL_HAdd, :LbyL_HArr, :LbyL_HArrft]
heuristic_pruning_methods = [:Flat_HAdd, :Flat_HArr, :LbyL_HAdd, :LbyL_HArr]
partial_pruning_methods = [:LbyLo_HAdd, :LbyLo_HArr]
greedy_pruning_methods = [:Flat_Gadd, :Flat_Garr, :LbyL_Gadd, :LbyL_Garr]
@testset "Checking integration with pruner" begin
    ds = specimen_sample()
    model = reflectinmodel(ds, d -> Dense(d, 4), SegmentedMean)

    @testset "heuristic methods" begin
        for e in [ConstExplainer(), StochasticExplainer(), GnnExplainer(200), GradExplainer(), ExplainMill.DafExplainer()]
            mk = ExplainMill.add_participation(stats(e, ds, model))

            o = softmax(model(ds).data)[:]
            τ = 0.9 * maximum(o) 
            class = argmax(softmax(model(ds).data)[:])
            f = () -> softmax(model(ds[mk]).data)[class] - τ
            @test !ExplainMill.flatsearch!(f, mk)
            @test !ExplainMill.levelbylevelsearch!(f, mk)

            @test !ExplainMill.flatsearch!(f, mk, random_removal = true)
            @test !ExplainMill.levelbylevelsearch!(f, mk, random_removal = true)

            # this test is more like a test that it passes
            for pruning_method in heuristic_pruning_methods
                @test explain(e, ds, model; rel_tol=0.9, pruning_method) isa ExplainMill.AbstractStructureMask
            end

            for pruning_method in partial_pruning_methods
                @test explain(e, ds, model; rel_tol=0.9, pruning_method) isa ExplainMill.AbstractStructureMask
            end
        end
    end

    @testset "non-heuristic methods" begin
        for e in [ConstExplainer()]
            mk = ExplainMill.add_participation(stats(e, ds, model))

            o = softmax(model(ds).data)[:]
            τ = 0.9 * maximum(o) 
            class = argmax(softmax(model(ds).data)[:])
            f = () -> softmax(model(ds[mk]).data)[class] - τ

            @test !ExplainMill.flatsfs!(f, mk, random_removal = true)
            @test !ExplainMill.levelbylevelsfs!(f, mk, random_removal = true)

            # this test is more like a test that it passes
            for pruning_method in greedy_pruning_methods
                @test explain(e, ds, model; rel_tol=0.9, pruning_method) isa ExplainMill.AbstractStructureMask
            end
        end
    end

    @testset "level-by-level seearch with partial evaluation" begin
        for e in [ConstExplainer(), StochasticExplainer(), GnnExplainer(200), GradExplainer(), ExplainMill.DafExplainer()]
            mk = ExplainMill.add_participation(stats(e, ds, model))

            o = softmax(model(ds).data)[:]
            τ = 0.9 * maximum(o) 
            class = argmax(softmax(model(ds).data)[:])
            f = (model, ds, mk) -> softmax(model(ds[mk]).data)[class] - τ

            @test !ExplainMill.levelbylevelsearch!(f, model, ds, mk, random_removal = true)
            @test !ExplainMill.prune!(mk, model, ds, class, 0.9 * logitconfgap(model, ds, class), :LbyLo_HAdd)
            @test !ExplainMill.prune!(mk, model, ds, class, 0.9 * logitconfgap(model, ds, class), :LbyLo_HArr)
        end
    end


end

@testset "Simple end2end test with a strict equality" begin
    ds = ArrayNode(ones(Float32, 6, 1))
    motif = [1, 0, 0, 0, 0, 1]
    model = ArrayModel(Dense(Float32.([1 0 0 0 0 1;-1 0 0 0 0 -1]), zeros(Float32, 2)))
    for e in [ConstExplainer(), StochasticExplainer(), GnnExplainer(200), GradExplainer(), ExplainMill.DafExplainer()]

        mk = ExplainMill.add_participation(stats(e, ds, model))
        class = [1]
        τ = logitconfgap(model, ds, class)
        f = () -> sum(logitconfgap(model, ds[mk], class) .- τ)

        ExplainMill.flatsfs!(f, mk, random_removal = true)
        @test prunemask(mk.mask) ≈ motif

        ExplainMill.levelbylevelsfs!(f, mk, random_removal = true)
        @test prunemask(mk.mask) ≈ motif

        for pruning_method in setdiff(ExplainMill.pruning_methods(), [:Flat_HAdd, :Flat_Gadd, :LbyL_HAdd, :LbyL_Gadd, :LbyLo_HAdd, :LbyLo_Gadd])
            @test prunemask(explain(e, ds, model; rel_tol=1, pruning_method).mask) ≈ motif
        end

        # Addition methods cannot remove useless parts of the explanation, therefore
        # we can only test that the explanation is a superset of a motif
        for pruning_method in [:Flat_HAdd, :Flat_Gadd, :LbyL_HAdd, :LbyL_Gadd, :LbyLo_HAdd, :LbyLo_Gadd]
            @test all(prunemask(explain(e, ds, model; rel_tol=1, pruning_method).mask) .≥ motif)
        end
    end
end