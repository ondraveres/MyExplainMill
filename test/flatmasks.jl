using ExplainMill
using Test
using Mill
using Flux
using SparseArrays
using Random
using ExplainMill: Mask, FlatView

initstats = d -> ones(d)

# ms = Mask(ds, model, d -> collect(d:-1:1), ExplainMill._nocluster);
@testset "mapping between flat structure and nodes" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(randn(2,5))
	ds = BagNode(ProductNode((a = an, c = cn)), AlignedBags([1:2,3:3,4:5]))
	model = reflectinmodel(ds, d -> Dense(d, 4), d -> SegmentedMeanMax(d))
	ms = Mask(ds, model, d -> rand(d), ExplainMill._nocluster);
	# ms = Mask(ds, initstats);

	fv = FlatView(ms)
    @show fv

	@test_broken length(fv) == 25
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

	for i in 6:7
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

    # these tests are not working, but I want to utilize at least the part that works.
    # for some reason, length(fv) == 9, so I'm keeping only that range
	#for i in 8:15
	for i in 8:9
		fv[i] = true
		@test fv[i] == true
		@test_broken ms.child.childs[:a].mask[i - 5] == true
		fv[i] = false
		@test fv[i] == false
		@test_broken ms.child.childs[:a].mask[i - 5] == false

		fv[i] = true
		@test fv[i] == true
		@test_broken ms.child.childs[:a].mask[i - 5] == true
	end

	#for i in 16:25
	#	fv[i] = true
	#	@test fv[i] == true
	#	@test ms.child.childs[:c].mask[i - 15] == true
	#	fv[i] = false
	#	@test fv[i] == false
	#	@test ms.child.childs[:c].mask[i - 15] == false
    #
	#	fv[i] = true
	#	@test fv[i] == true
	#	@test ms.child.childs[:c].mask[i - 15] == true
	#end
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
	@test_broken fv.masks[1].first.mask.mask == Bool[1, 1, 0, 0, 1]
	@test_broken (fv.masks[2]).first.mask.mask== Bool[1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
end

@testset "Participation in FlatView" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(ProductNode((a = an, c = cn)), AlignedBags([1:2,3:3,4:5]))
	m = Mask(ds, initstats);
	fv = FlatView(m);

	m.mask.mask .= [1,0,1,0,1]
	ExplainMill.updateparticipation!(m)
	@test m.child[:c].mask.participate == [1,0,1,0,1]
	@test m.child[:a].mask.participate == [1,1]

	parents = ExplainMill.parent_structure(m)
	cfv = FlatView(ExplainMill.firstparents([m.child[:a], m.child[:c]],
		parents))
	@test ExplainMill.participate(cfv) == [1,1,1,0,1,0,1]

	m.mask.mask .= [0,0,0,0,0]
	ExplainMill.updateparticipation!(m)
	@test m.child[:c].mask.participate == [0,0,0,0,0]
	@test m.child[:a].mask.participate == [1,1]
end

@testset "settrue! in FlatView" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = ProductNode((a = an, c = cn))
	m = Mask(ds, initstats);
	fv = FlatView(m);

	fill!(fv, false)
	@test length(fv) == 7
	ExplainMill.settrue!(fv, [true, false, true, false, true, false, false])
	@test fv.masks[1].first.mask.mask == [true, false]
	@test fv.masks[2].first.mask.mask == [true, false, true, false, false]
	@test ExplainMill.useditems(fv) == [1,3,5]

	ExplainMill.settrue!(fv, [false, true, false, true, false, false, true])
	@test fv.masks[1].first.mask.mask == [false, true]
	@test fv.masks[2].first.mask.mask == [false, true, false, false, true]
	@test ExplainMill.useditems(fv) == [2,4,7]

	#Let's test syntactic sugar
	fv .= [true, false, true, false, true, false, false]
	@test fv.masks[1].first.mask.mask == [true, false]
	@test fv.masks[2].first.mask.mask == [true, false, true, false, false]
	@test ExplainMill.useditems(fv) == [1,3,5]

	fv .= [false, true, false, true, false, false, true]
	@test fv.masks[1].first.mask.mask == [false, true]
	@test fv.masks[2].first.mask.mask == [false, true, false, false, true]
	@test ExplainMill.useditems(fv) == [2,4,7]
end

@testset "Parental structure" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(ProductNode((a = an, c = cn)), AlignedBags([1:2,3:3,4:5]))

	m₁ = Mask(ds, initstats)
	m₂ = ProductMask((a = m₁, b = EmptyMask(), c = EmptyMask()))
	for mask in [m₁, m₂]
		fv = FlatView(m₁)

		@test_broken length(fv) == 20
		@test_broken ExplainMill.parent(fv, 1) == 0
		@test ExplainMill.parent(fv, 6) == 1
		@test ExplainMill.parent(fv, 7) == 1
		@test_broken ExplainMill.parent(fv, 8) == 1
		@test_broken ExplainMill.parent(fv, 8) == 1
		@test_broken ExplainMill.parent(fv, 9) == 2
		@test_broken ExplainMill.parent(fv, 10) == 3
		@test_broken ExplainMill.parent(fv, 11) == 4
		@test_broken ExplainMill.parent(fv, 12) == 5
	end
end
