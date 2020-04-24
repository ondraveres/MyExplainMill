using Test, ExplainMill, Mill, SparseArrays
using ExplainMill: FlatView, index_in_parent
using ExplainMill: Mask, ProductMask, CategoricalMask, NGramMatrixMask, BagMask, EmptyMask
"""
	verifies that all parents are BagMasks (ProductMasks are igonored)
"""

function testparentship(masks)
	ns, ii = [v.first for v in masks],[v.second for v in masks]
	ii = filter(i -> i != 0, ii)
	all(map(x -> isa(x, ExplainMill.BagMask), ns[ii]))
end

@testset "mapping between flat structure and nodes" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(randn(2,5))
	ds = BagNode(ProductNode((a = an, c = cn)), AlignedBags([1:2,3:3,4:5]))

	mask = Mask(ds)
	fv = FlatView(mask)

	@test length(fv) == 9
	for i in 1:5
		fv[i] = true
		@test fv[i] == true
		@test mask.mask.mask[i] == true
		fv[i] = false
		@test fv[i] == false
		@test mask.mask.mask[i] == false

		fv[i] = true
		@test fv[i] == true
		@test mask.mask.mask[i] == true
	end

	for i in 6:7
		fv[i] = true
		@test fv[i] == true
		@test mask.child.childs[:a].mask[i - 5] == true
		fv[i] = false
		@test fv[i] == false
		@test mask.child.childs[:a].mask[i - 5] == false

		fv[i] = true
		@test fv[i] == true
		@test mask.child.childs[:a].mask[i - 5] == true
	end

	for i in 8:9
		fv[i] = true
		@test fv[i] == true
		@test mask.child.childs[:c].mask[i - 7] == true
		fv[i] = false
		@test fv[i] == false
		@test mask.child.childs[:c].mask[i - 7] == false

		fv[i] = true
		@test fv[i] == true
		@test mask.child.childs[:c].mask[i - 7] == true
	end
end

@testset "index in parent" begin
	m = Mask(ArrayNode(reshape(collect(1:10), 2, 5)))
	@test index_in_parent(m,1) == 1
	@test index_in_parent(m,2) == 1

	m = Mask(ArrayNode(sparse([1 0 3 0 5; 2 2 0 4 3])))
	@test index_in_parent(m,1) == 1
	@test index_in_parent(m,2) == 1
	@test index_in_parent(m,3) == 2
	@test index_in_parent(m,4) == 3
	@test index_in_parent(m,7) == 5

	m = Mask(BagNode(ArrayNode(reshape(collect(1:10), 2, 5)), AlignedBags([1:2,3:3,4:5])))
	@test index_in_parent(m,1) == 1
	@test index_in_parent(m,2) == 1
	@test index_in_parent(m,3) == 2
	@test index_in_parent(m,4) == 3
	@test index_in_parent(m,5) == 3
end


@testset "Parental structure" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(ProductNode((a = an, c = cn)), AlignedBags([1:2,3:3,4:5]))

	m₁ = Mask(ds)
	m₂ = ProductMask((a = m₁, b = EmptyMask(), c = EmptyMask()))
	for mask in [m₁, m₂]
		fv = FlatView(mask)

		@test length(fv) == 12
		@test ExplainMill.parent(fv, 1) == 0
		@test ExplainMill.parent(fv, 6) == 1
		@test ExplainMill.parent(fv, 7) == 1
		@test ExplainMill.parent(fv, 8) == 1
		@test ExplainMill.parent(fv, 8) == 1
		@test ExplainMill.parent(fv, 9) == 2
		@test ExplainMill.parent(fv, 10) == 3
		@test ExplainMill.parent(fv, 11) == 4
		@test ExplainMill.parent(fv, 12) == 5
	end
end
