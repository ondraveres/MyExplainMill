@testset "testing infering of sample membership" begin
	sn = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
	ds = BagNode(sn, AlignedBags([1:2,3:3,4:5]))
	pm = Mask(ds, d -> rand(d))
	ExplainMill.updatesamplemembership!(pm, nobs(ds))
	@test pm.mask.outputid ≈ [1, 1, 2, 3, 3]
end