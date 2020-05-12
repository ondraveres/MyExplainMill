using Test, ExplainMill, Mill, SparseArrays, Flux, Random, StatsBase
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

initstats = d -> ones(d)

@testset "mapping between flat structure and nodes" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(randn(2,5))
	ds = BagNode(ProductNode((a = an, c = cn)), AlignedBags([1:2,3:3,4:5]))
	ms = Mask(ds, initstats);

	fv = FlatView(ms)

	@test length(fv) == 25
	for i in 1:5
		fv[i] = true
		@test fv[i] == true
		@test ms.mask.mask[i] == true
		fv[i] = false
		@test fv[i] == false
		@test ms.mask.mask[i] == false

		fv[i] = true
		@test fv[i] == true
		@test ms.mask.mask[i] == true
	end

	for i in 6:15
		fv[i] = true
		@test fv[i] == true
		@test ms.child.childs[:a].mask[i - 5] == true
		fv[i] = false
		@test fv[i] == false
		@test ms.child.childs[:a].mask[i - 5] == false

		fv[i] = true
		@test fv[i] == true
		@test ms.child.childs[:a].mask[i - 5] == true
	end

	for i in 16:25
		fv[i] = true
		@test fv[i] == true
		@test ms.child.childs[:c].mask[i - 15] == true
		fv[i] = false
		@test fv[i] == false
		@test ms.child.childs[:c].mask[i - 15] == false

		fv[i] = true
		@test fv[i] == true
		@test ms.child.childs[:c].mask[i - 15] == true
	end
end

@testset "fill! in FlatView" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(ProductNode((a = an, c = cn)), AlignedBags([1:2,3:3,4:5]))
	m = Mask(ds, initstats);
	fv = FlatView(m);

	fill!(fv, false)
	@test !any(fv[i] for i in 1:length(fv))
	fill!(fv, true)
	@test all(fv[i] for i in 1:length(fv))

	Random.seed!(0)
	# map(m -> sample!(m.mask), fv)
	map(m -> sample!(m.mask), fv)
	@test fv.masks[1].first.mask.mask == Bool[1, 1, 0, 0, 1]
	@test (fv.masks[2]).first.mask.mask== Bool[1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
end
@testset "index in parent" begin
	ds = ArrayNode(reshape(collect(1:10), 2, 5))
	m = Mask(ds, initstats);
	@test index_in_parent(m,1) == 1
	@test index_in_parent(m,2) == 1
	@test index_in_parent(m,3) == 2
	@test index_in_parent(m,4) == 2

	ds = ArrayNode(sparse([1 0 3 0 5; 2 2 0 4 3]))
	m = Mask(ds, initstats);
	@test index_in_parent(m,1) == 1
	@test index_in_parent(m,2) == 1
	@test index_in_parent(m,3) == 2
	@test index_in_parent(m,4) == 3
	@test index_in_parent(m,7) == 5

	ds = BagNode(ArrayNode(reshape(collect(1:10), 2, 5)), AlignedBags([1:2,3:3,4:5]))
	m = Mask(ds, initstats);
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

	m₁ = Mask(ds, initstats)
	m₂ = ProductMask((a = m₁, b = EmptyMask(), c = EmptyMask()))
	for mask in [m₁, m₂]
		fv = FlatView(m₁)

		@test length(fv) == 20
		@test_broken ExplainMill.parent(fv, 1) == 0
		@test_broken ExplainMill.parent(fv, 6) == 1
		@test_broken ExplainMill.parent(fv, 7) == 1
		@test_broken ExplainMill.parent(fv, 8) == 1
		@test_broken ExplainMill.parent(fv, 8) == 1
		@test_broken ExplainMill.parent(fv, 9) == 2
		@test_broken ExplainMill.parent(fv, 10) == 3
		@test_broken ExplainMill.parent(fv, 11) == 4
		@test_broken ExplainMill.parent(fv, 12) == 5
	end
end
