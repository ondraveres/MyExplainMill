using ExplainMill, Mill, SparseArrays
using ExplainMill: prune, Mask, invalidate!
using Setfield
using Test

@testset "Testing correctness of detecting samples that should not be considered in the calculation of daf values" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(sparse([1 0 3 0 5; 1 2 0 4 0]))

	mask = Mask(cn)
	invalidate!(mask, [1,2])
	@test mask.participate ≈ [false, false, false, true, true, true]
	mask.participate .= true
	invalidate!(mask, [2,4])
	@test mask.participate ≈ [true, true, false, true, false, true]


	sn = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
	mask = Mask(sn)
	invalidate!(mask, [2,4])
	@test mask.participate ≈ [true, false, true, false, true]

	ds = BagNode(sn, AlignedBags([1:2,3:3,4:5]))
	mask = Mask(ds)
	invalidate!(mask, [1,3])
	@test mask.participate ≈ mask.child.participate ≈ [false, false, true, false, false]
	mask = Mask(ds)
	mask.mask[[1,2,4,5]] .= false
	invalidate!(mask)
	@test mask.mask ≈ [false, false, true, false, false]
	@test all(mask.participate .== true)
	@test mask.child.participate ≈ [false, false, true, false, false]


	ds = BagNode(BagNode(sn, AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))
	mask = Mask(ds)
	invalidate!(mask, [1])
	@test all(mask.mask .== true)
	@test all(mask.participate .== false)
	@test all(mask.child.mask .== true)
	@test all(mask.child.participate .== false)
	@test all(mask.child.child.mask .== true)
	@test all(mask.child.child.participate .== false)
	mask = Mask(ds)
	mask.mask[[1,3]] .= false
	invalidate!(mask)
	@test mask.mask ≈ [false, true, false]
	@test all(mask.participate .== true)
	@test mask.child.participate ≈ [false, false, true, false, false]
	@test mask.child.mask .& mask.child.participate  ≈ [false, false, true, false, false]

	ds = BagNode(TreeNode((a = cn, b = sn)), AlignedBags([1:2,3:3,4:5]))
	mask = Mask(ds)
	mask.mask[[1,3]] .= false
	invalidate!(mask)
	@test mask.mask ≈ [false, true, false, true, true]
	@test all(mask.participate .== true)
	@test mask.child.childs[:a].participate ≈ [false, false, true, false, true, true]
	@test mask.child.childs[:b].participate ≈ [false, true, false, true, true]
	mask = Mask(ds)
	invalidate!(mask,[1,3])
	@test all(mask.mask .== true)
	@test mask.participate ≈ [false, false, true, false, false]
	@test mask.child.childs[:a].participate ≈ [false, false, false, true, false, false]
	@test mask.child.childs[:b].participate ≈ [false, false, true, false, false]
end


@testset "Testing prunning of samples " begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(BagNode(TreeNode((a = an, c = cn)), AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))

	mask = BagMask(
			BagMask(
				TreeMask((a = ArrayMask([true,false]),
				c = ArrayMask([true, true, true, false, true]),)
				),
			[true,false,true,false,true]),
			[true,true,true])
	dss = prune(ds, mask)

	@test nobs(dss) == 1
	@test nobs(dss.data) == 3
	@test nobs(dss.data.data) == 3
	@test dss.data.data.data.c.data.nzval ≈ [1,3,5]
	@test dss.data.data.data.a.data ≈ [1 5 9; 0 0 0 ]


	mask = BagMask(
			BagMask(
				TreeMask((a = ArrayMask([false,true]),
				c = ArrayMask([false, true, false, true, false]),)
				),
			[true,false,true,false,true]),
			[true,true,true])
	dss = prune(ds, mask)

	@test nobs(dss) == 1
	@test nobs(dss.data) == 3
	@test nobs(dss.data.data) == 3
	@test dss.data.data.data.c.data.nzval ≈ [0, 0, 0]
	@test dss.data.data.data.a.data ≈ [0 0 0; 2 6 10]

	mask = BagMask(
			BagMask(
				TreeMask((a = ArrayMask([false,true]),
				c = ArrayMask([true, true, true, true, true]),)
				),
			[true,false,false,false,true]),
			[true,true,true])
	dss = prune(ds, mask)

	@test nobs(dss) == 1
	@test nobs(dss.data) == 3
	@test all(dss.data.bags.bags .== [1:1, 0:-1, 2:2])
	@test nobs(dss.data.data) == 2
	@test dss.data.data.data.c.data.nzval ≈ [1, 5]
	@test dss.data.data.data.a.data ≈ [0 0; 2 10]
end


@testset "Generation of TreeDaf structure" begin 
	x = reshape(collect(1:10), 2, 5)
	an = ArrayNode(x)
	bn = BagNode(deepcopy(an), AlignedBags([1:2,3:5]))
	cn = ArrayNode(sprand(4, 2, 0.3))
	tn = TreeNode((a = an[1:2], b = bn, c = cn))

	daf = Daf(tn);
	@test typeof(daf.childs.a) <: ArrayDaf{Duff.Daf}
	@test typeof(daf.childs.b) <: BagDaf{ArrayDaf{Daf},AlignedBags,Daf}
	@test typeof(daf.childs.b.child) <: ArrayDaf{Daf}
	@test typeof(daf.childs.b.daf) <: Daf
	@test typeof(daf.childs.c) <: SparseArrayDaf{Duff.Daf}

	@test length(Daf(cn).daf) == nnz(cn.data)
	@test length(Daf(an).daf) == size(an.data,1)
	@test length(Daf(bn).daf) == nobs(bn.data)
end


@testset "Sampling of basic structures for explanation" begin
	an = ArrayNode(rand(2,5))
	bn = BagNode(deepcopy(an), AlignedBags([1:2,3:5]))
	cn = ArrayNode(sprand(4, 2, 0.3))
	tn = TreeNode((a = an[1:2], b = bn, c = cn))

	ann, mask = sample(Daf(an), an)
	@test ann.data[mask.mask, :] ≈ an.data[mask.mask, :]
	@test all(ann.data[.!mask.mask, :] .== 0)

	abn, mask = sample(Daf(bn), bn)
	@test nobs(abn.data) == sum(mask.mask)
	@test abn.data.data[mask.child_masks.mask, :] ≈ bn.data.data[:, mask.mask][mask.child_masks.mask, :]
	@test all(abn.data.data[.!mask.child_masks.mask, :] .== 0)


	cnn, mask = sample(Daf(cn), cn)
	@test cnn.data.nzval[mask.mask] ≈ cn.data.nzval[mask.mask]
	@test all(cnn.data.nzval[.!mask.mask, :] .== 0)

	abn, mask = sample(Daf(bn), bn)
	@test nobs(abn.data) == sum(mask.mask)
	@test abn.data.data[mask.child_masks.mask, :] ≈ bn.data.data[:, mask.mask][mask.child_masks.mask, :]
	@test all(abn.data.data[.!mask.child_masks.mask, :] .== 0)

	atn, mask = sample(Daf(tn), tn)
	@test atn.data.a.data[mask.child_masks.a.mask,:] ≈ tn.data.a.data[mask.child_masks.a.mask,:]
	@test all(atn.data.a.data[.!mask.child_masks.a.mask,:] .== 0)
	@test nobs(atn.data.b.data) == sum(mask.child_masks.b.mask)
	@test atn.data.b.data.data[mask.child_masks.b.child_masks.mask, :] ≈ bn.data.data[:, mask.child_masks.b.mask][mask.child_masks.b.child_masks.mask, :]
	@test all(atn.data.b.data.data[.!mask.child_masks.b.child_masks.mask, :] .== 0)
	@test atn.data.c.data.nzval[mask.child_masks.c.mask] ≈ cn.data.nzval[mask.child_masks.c.mask]
	@test all(atn.data.c.data.nzval[.!mask.child_masks.c.mask, :] .== 0)
end

@testset "Update of Daf" begin 
	an = ArrayNode(rand(2,5))
	bn = BagNode(deepcopy(an), AlignedBags([1:2,3:5]))
	cn = ArrayNode(sprand(4, 2, 0.3))
	tn = TreeNode((a = an[1:2], b = bn, c = cn))

	daf = Daf(tn);
	mask = sample(daf, tn)[2]
	Duff.update!(daf, mask, 0.5)

	ss = daf.childs.a.daf;
	m = mask.child_masks.a.mask;
	@test all(ss.absent.n[.!m] .== 1)
	@test all(ss.absent.n[m] .== 0)
	@test all(ss.present.n[m] .== 1)
	@test all(ss.present.n[.!m] .== 0)

	ss = daf.childs.b.daf;
	m = mask.child_masks.b.mask;
	@test all(ss.absent.n[.!m] .== 1)
	@test all(ss.absent.n[m] .== 0)
	@test all(ss.present.n[m] .== 1)
	@test all(ss.present.n[.!m] .== 0)

	ss = daf.childs.b.child.daf;
	m = mask.child_masks.b.child_masks.mask;
	@test all(ss.absent.n[.!m] .== 1)
	@test all(ss.absent.n[m] .== 0)
	@test all(ss.present.n[m] .== 1)
	@test all(ss.present.n[.!m] .== 0)

	ss = daf.childs.c.daf;
	m = mask.child_masks.c.mask;
	@test all(ss.absent.n[.!m] .== 1)
	@test all(ss.absent.n[m] .== 0)
	@test all(ss.present.n[m] .== 1)
	@test all(ss.present.n[.!m] .== 0)
end

@testset "Nesting of bagnodes." begin
	an = ArrayNode(rand(2,5))
	bn = BagNode(deepcopy(an), AlignedBags([1:2,3:5]))
	bnn = BagNode(bn, AlignedBags([1:2]))
	daf = Daf(bnn);
	mask = BagMask(BagMask(ArrayMask(Bool[1, 0]), Bool[0, 1, 0]), Bool[0, 1])
	Duff.update!(daf, mask, 0.5)

	ss = daf.daf;
	m = mask.mask;
	@test all(ss.absent.n[.!m] .== 1)
	@test all(ss.absent.n[m] .== 0)
	@test all(ss.present.n[m] .== 1)
	@test all(ss.present.n[.!m] .== 0)

	ss = daf.child.daf;
	m = mask.child_masks.mask;
	valid_columns = vcat(collect.(bnn.data.bags[mask.mask])...)
	@test all(ss.absent.n[valid_columns[.!m]] .== 1)
	@test all(ss.absent.n[valid_columns[m]] .== 0)
	@test all(ss.present.n[valid_columns[m]] .== 1)
	@test all(ss.present.n[valid_columns[.!m]] .== 0)

	invalid_columns = setdiff(vcat(collect.(bnn.data.bags)...), valid_columns)
	@test all(ss.absent.n[invalid_columns] .== 0)
	@test all(ss.present.n[invalid_columns] .== 0)
end

@testset "Handling of Sparse Arrays with subset of columns" begin 
	x = ArrayNode(sparse([1 1 0 0 2.0; 0 1 0 1 0.0]))
	daf = Daf(x)
	mask = ArrayMask(Bool[1, 0, 1])
	valid_columns = [2,5]
	Duff.update!(daf, mask, 0.5, valid_columns)
	
	ss = daf.daf
	m = mask.mask
	valid_columns = [2,3,5]
	@test all(ss.absent.n[valid_columns[.!m]] .== 1)
	@test all(ss.absent.n[valid_columns[m]] .== 0)
	@test all(ss.absent.s[valid_columns[m]] .== 0)
	@test all(ss.absent.s[valid_columns[.!m]] .== 0.5)
	@test all(ss.present.n[valid_columns[m]] .== 1)
	@test all(ss.present.n[valid_columns[.!m]] .== 0)
	@test all(ss.present.s[valid_columns[m]] .== 0.5)
	@test all(ss.present.s[valid_columns[.!m]] .== 0)
end

@testset "Testing the daf framework" begin 
	x = Float32.(reshape(collect(1:10), 2, 5))
	an = ArrayNode(x)
	bn = BagNode(deepcopy(an), AlignedBags([1:2,3:5]))
	tn = TreeNode((a = an[1:2], b = bn))
	dss = BagNode(tn, AlignedBags([1:2]))

	for ds in [dss, tn[1], bn[1], an[1]]
		daf = Duff.Daf(ds);
		model = reflectinmodel(ds, d -> Dense(d,2), d -> SegmentedMean(d), b = Dict("" => d -> Dense(d,1)))
		trials = 1000
		for i in 1:trials
			dss, mask = sample(daf, ds)
			v = model(dss).data[1]
			Duff.update!(daf, mask, v)
		end
		@test true
	end
end

@testset "Testing the extraction of daf stats" begin 
	x = Float32.(reshape(collect(1:10), 2, 5))
	an = ArrayNode(x)
	bn = BagNode(deepcopy(an), AlignedBags([1:2,3:5]))
	tn = TreeNode((a = an[1:2], b = bn))
	ds = BagNode(tn, AlignedBags([1:2]))

	daf = Duff.Daf(ds);
	model = reflectinmodel(ds, d -> Dense(d,2), d -> SegmentedMean(d), b = Dict("" => d -> Dense(d,1)))


	mask, dafs = ExplainMill.masks_and_stats(daf)
	dafs = sort(dafs, lt = (x,y) -> x.depth < y.depth)
	@test all([length(dafs[i].m) == length(dafs[i].d) for i in 1:length(dafs)])
	@test all([all(dafs[i].m) for i in 1:length(dafs)])
	@test mask.mask[1] == true
	dafs[1].m[1] = false
	@test mask.mask[1] == false
end
