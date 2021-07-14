using ExplainMill
using Mill
using Test
using Flux
using SparseArrays
using StatsBase: nobs
using ExplainMill: collectmasks

using ExplainMill: DafMask, create_mask_structure, ParticipationTracker, updateparticipation!

@testset "Check that heuristic values are non-zeros" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	on = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	sn = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 256, 2053))
	ds = BagNode(BagNode(ProductNode((;an, on, cn, sn)), AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))
	model = reflectinmodel(ds, d -> Dense(d, 4), SegmentedMean)

	for e in [ConstExplainer(), StochasticExplainer(), GradExplainer(), GnnExplainer(200), ExplainMill.DafExplainer()]
		mk = stats(e, ds, model)
		@test all(sum(abs.(heuristic(m))) > 0 for m in collectmasks(mk))
	end
end
