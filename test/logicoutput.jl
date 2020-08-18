using ExplainMill, Mill, JsonGrinder
using ExplainMill: Mask, yarason, participate, prunemask, addor, OR
using Test

@testset "OR relationship" begin 
	xs = ["a","b","c","d","e"]
	m = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))

	@test addor(m, xs[participate(m) .& prunemask(m)]) == [OR(["a", "e"]), OR(["b", "d"]), ["c"], OR(["b", "d"]), OR(["a", "e"]),]
	m.mask[1] = false
	@test addor(m, xs[participate(m) .& prunemask(m)]) == [OR(["b", "d"]), ["c"], OR(["b", "d"])]
	m.mask[1:2] = [true,false]
	@test addor(m, xs[participate(m) .& prunemask(m)]) == [OR(["a", "e"]), ["c"], OR(["a", "e"]),]
	m.mask[2:3] = [true, false]
	@test addor(m, xs[participate(m) .& prunemask(m)]) == [OR(["a", "e"]), OR(["b", "d"]), OR(["b", "d"]), OR(["a", "e"])]

	m = ExplainMill.Mask(5, d -> zeros(d))
	m.mask[1] = false
	@test addor(m, xs[participate(m) .& prunemask(m)]) == xs[participate(m) .& prunemask(m)]
end

@testset "Testing correctness of detecting samples that should not be considered in the calculation of daf values" begin
	an = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
	am = Mask(an, d -> rand(d))
	m, ds = am, an
	all(yarason(an, am, nothing) .== ["a","b","c","d","e"])
	all(yarason(an, am, nothing) .== ["a","b","c","d","e"])

	cn = ArrayNode(sparse([1 0 3 0 5; 1 2 0 4 0]))
	
	ds = BagNode(BagNode(sn, AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))
	ds = BagNode(ProductNode((a = cn, b = sn)), AlignedBags([1:2,3:3,4:5]))
end


@testset "mapmask" begin
	an = Mask(ArrayNode(reshape(collect(1:10), 2, 5)), d -> rand(d))
	mapmask(an) do m
		participate(m)[1] = false
		prunemask(m)[2] = false
	end
	@test prunemask(an) ≈ [true, false]
	@test participate(an) ≈ [false, true]


	an = Mask(ArrayNode(sparse([1 0 3 0 5; 1 2 0 4 0])), d -> rand(d))
	mapmask(an) do m
		participate(m)[1] = false
		prunemask(m)[2] = false
	end
	@test prunemask(an) ≈ [true, false, true, true, true, true]
	@test participate(an) ≈ [false, true, true, true, true, true]

	an = Mask(ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4)), d -> rand(d))
	mapmask(an) do m
		participate(m)[1] = false
		prunemask(m)[2] = false
	end
	@test prunemask(an) ≈ [true, false, true, true, true]
	@test participate(an) ≈ [false, true, true, true, true]


	an = Mask(ArrayNode(NGramMatrix(["a","b","c"],3,256,2053)), d -> rand(d))
	mapmask(an) do m
		participate(m)[1] = false
		prunemask(m)[2] = false
	end
	@test prunemask(an) ≈ [true, false, true]
	@test participate(an) ≈ [false, true, true]

	an = Mask(ProductNode((a = ArrayNode(NGramMatrix(["a","b","c","d","e"],3,256,2053)), 
		b = ArrayNode(reshape(collect(1:10), 2, 5)))), d -> rand(d))
	mapmask(an) do m
		participate(m)[1] = false
		prunemask(m)[2] = false
	end
	@test prunemask(an.childs.a) ≈ [true, false, true, true, true]
	@test participate(an.childs.a) ≈ [false, true, true, true, true]
	@test prunemask(an.childs.b) ≈ [true, false]
	@test participate(an.childs.b) ≈ [false, true]
	@test participate(an) == nothing
	@test prunemask(an) == nothing


	an = Mask(BagNode(ArrayNode(reshape(collect(1:10), 2, 5)), AlignedBags([1:2,3:3,4:5])), d -> rand(d))
	mapmask(an) do m
		participate(m)[1] = false
		prunemask(m)[2] = false
	end
	@test prunemask(an) ≈ [true, false, true, true, true,]
	@test participate(an) ≈ [false, true, true, true, true]
	@test prunemask(an.child) ≈ [true, false]
	@test participate(an.child) ≈ [false, true]
end


@testset "Testing prunning of samples " begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	on = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(BagNode(ProductNode((a = an, c = cn, o = on)), AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))

	m = BagMask(
			BagMask(
				ProductMask((a = MatrixMask([true,false], 5),
				c = SparseArrayMask([true, true, true, false, true], [1, 2, 3, 4, 5]),
				o = CategoricalMask([true, true, true, false, false]),)
				), ds.bags.bags,
			[true,false,true,false,true]),
			ds.bags,
			[true,true,true])
	dss = ds[m]

	@test nobs(dss) == 1
	@test nobs(dss.data) == 3
	@test nobs(dss.data.data) == 3
	@test dss.data.data.data.a.data ≈ [1 5 9; 0 0 0 ]
	@test dss.data.data.data.c.data.nzval ≈ [1,3,5]
	@test dss.data.data.data.o.data ≈ Flux.onehotbatch([1,3,4], 1:4)
	@test ds[m] == prune(ds, m)

	m = BagMask(
		BagMask(
			ProductMask((a = MatrixMask([false,true], 5),
			c = SparseArrayMask([false, true, false, true, false], [1, 2, 3, 4, 5]),
			o = CategoricalMask([false, true, false, true, false]),)
			), ds.bags.bags,
		[true,false,true,false,true]),
		ds.bags,
		[true,true,true])
	dss = ds[m]

	@test nobs(dss) == 1
	@test nobs(dss.data) == 3
	@test nobs(dss.data.data) == 3
	@test dss.data.data.data.c.data.nzval ≈ [0, 0, 0]
	@test dss.data.data.data.o.data ≈ Flux.onehotbatch([4,4,4], 1:4)
	@test dss.data.data.data.a.data ≈ [0 0 0; 2 6 10]
	@test ds[m] == prune(ds, m)

	m = BagMask(
		BagMask(
			ProductMask((a = MatrixMask([false,true], 5),
			c = SparseArrayMask([true, true, true, true, true], [1, 2, 3, 4, 5]),
			o = CategoricalMask([false, true, false, true, false]),)
			), ds.bags.bags,
		[true,false,false,false,true]),
		ds.bags,
		[true,true,true])
	dss = ds[m]

	@test nobs(dss) == 1
	@test nobs(dss.data) == 3
	@test all(dss.data.bags.bags .== [1:1, 0:-1, 2:2])
	@test nobs(dss.data.data) == 2
	@test dss.data.data.data.c.data.nzval ≈ [1, 5]
	@test dss.data.data.data.o.data ≈ Flux.onehotbatch([4,4], 1:4)
	@test dss.data.data.data.a.data ≈ [0 0; 2 10]
	@test ds[m] == prune(ds, m)


	@test ds[ExplainMill.EmptyMask()] == ds

	a = ArrayNode(NGramMatrix(["a","b","c","d","e"],3,256,2053))
	an = Mask(a, d -> rand(d))
	prunemask(an) .= [true, false, true, false, true]
	@test a[an].data.s == ["a", "", "c", "", "e"]
	@test a[an,[true,false,true,true,false]].data.s == ["a", "c", ""]

	bs = BagNode(a, AlignedBags([1:2,3:3,4:5]))
	ms = Mask(bs, d -> rand(d))
	@test nobs(bs[ms, fill(false, 3)]) == 0
	@test nobs(bs[ms, [true,false,true]]) == 2
	@test nobs(bs[ExplainMill.EmptyMask(), [true,false,true]]) == 2
end

@testset "testing infering of sample membership" begin
	sn = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
	ds = BagNode(sn, AlignedBags([1:2,3:3,4:5]))
	pm = Mask(ds, d -> rand(d))
	ExplainMill.updatesamplemembership!(pm, nobs(ds))
	@test pm.mask.outputid ≈ [1, 1, 2, 3, 3]
end

@testset "remapping the cluster" begin
	@test ExplainMill.normalize_clusterids([2,3,2,3,4]) ≈ [1,2,1,2,3]
	@test ExplainMill.normalize_clusterids([1,2,2,1]) ≈ [1,2,2,1]
end

@testset "print masks" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	on = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(BagNode(ProductNode((a = an, c = cn, o = on)), AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))

	m = ExplainMill.BagMask(
			ExplainMill.BagMask(
				ExplainMill.ProductMask((a = ExplainMill.MatrixMask([true, false], 5),
				c = ExplainMill.SparseArrayMask([true, true, true, false, true], [1, 2, 3, 4, 5]),
				o = ExplainMill.CategoricalMask([true, true, true, false, false]),)
				), ds.bags.bags,
			[true,false,true,false,true]),
			ds.bags,
			[true,true,true])

    @test_broken begin
        # buf = IOBuffer()
        # printtree(buf, m, trav=true)
        # str_repr = String(take!(buf))
        str_repr = ""
        str_repr ==
        """
        BagMask [""]
        └── BagMask ["U"]
        └── ProductMask ["k"]
        ├── a: MatrixMask ["o"]
        ├── c: SparseArrayMask ["s"]
        └── o: CategoricalMask ["w"]"""
    end
end
