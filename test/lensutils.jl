using Test
using ExplainMill: findnonempty, ModelLens

@testset "findnonempty" begin
	@test findnonempty(ArrayNode(randn(2,0))) == nothing
	@test findnonempty(ArrayNode(randn(2,1))) == [(@lens _.data)]
	@test findnonempty(BagNode(ArrayNode(randn(2,0)), [0:-1])) == nothing
	@test findnonempty(BagNode(ArrayNode(randn(2,1)), [1:1])) == [(@lens _.data.data)]
	@test findnonempty(ProductNode((a = BagNode(ArrayNode(randn(2,0)), [0:-1]), ))) == nothing
	@test findnonempty(ProductNode((a = BagNode(ArrayNode(randn(2,1)), [1:1]), ))) == [(@lens _.data.a.data.data)]
	@test findnonempty(ProductNode((a = BagNode(ArrayNode(randn(2,1)), [1:1]), b = ArrayNode(randn(2,1))))) == [(@lens _.data.a.data.data), (@lens _.data.b.data)]
	@test findnonempty(ProductNode((a = BagNode(ArrayNode(randn(2,0)), [0:-1]), b = ArrayNode(randn(2,1))))) == [(@lens _.data.b.data)]
end

@testset "ModelLens" begin
	model = ArrayModel(Dense(2,2))
	@test ModelLens(model, @lens _.data) == @lens _.m
	model = BagModel(ArrayModel(Dense(2,2)), SegmentedMean(2), Dense(2,2))
	@test ModelLens(model, @lens _.data) == @lens _.im
	@test ModelLens(model , @lens _.data.data) == @lens _.im.m
	model = ProductModel((
			a = BagModel(ArrayModel(Dense(2,2)), SegmentedMean(2), Dense(2,2)),
			b = ArrayModel(Dense(2,2)),
			))
	@test ModelLens(model, @lens _.data) == @lens _.ms
	@test ModelLens(model, @lens _.data.a) == @lens _.ms.a
	@test ModelLens(model, @lens _.data.a.data) == @lens _.ms.a.im
	@test ModelLens(model, @lens _.data.a.data.data) == @lens _.ms.a.im.m
	@test ModelLens(model, @lens _.data.b) == @lens _.ms.b
	@test ModelLens(model, @lens _.data.b.data) == @lens _.ms.b.m	
end

