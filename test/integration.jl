using ExplainMill
using Mill
using Test
using Flux
using SparseArrays
using StatsBase: nobs
using ExplainMill: collectmasks

@testset "Check that heuristic values are non-zeros" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	on = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(BagNode(ProductNode((a = an, c = cn, o = on)), AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))
	model = reflectinmodel(ds)

	for e in [ConstExplainer(), StochasticExplainer(), GradExplainer(), GnnExplainer(200)]
		mk = stats(e, ds, model)
		@test all(sum(heuristic(m)) > 0 for m in collectmasks(mk))
	end
end
