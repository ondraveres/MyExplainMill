ExplainMill.Mask(m::Vector{Bool}) = ExplainMill.Mask(m, fill(true, length(m)), fill(0, length(m)), ones(length(m)), nothing)
ExplainMill.MatrixMask(m::Vector{Bool}, d) = ExplainMill.MatrixMask(ExplainMill.Mask(m), length(m), d)
ExplainMill.CategoricalMask(m::Vector{Bool}) = ExplainMill.CategoricalMask(ExplainMill.Mask(m))
ExplainMill.BagMask(child, bags, m::Vector{Bool}) = ExplainMill.BagMask(child, bags, ExplainMill.Mask(m))
ExplainMill.NGramMatrixMask(m::Vector{Bool}) = ExplainMill.NGramMatrixMask(ExplainMill.Mask(m))
ExplainMill.SparseArrayMask(m::Vector{Bool}, columns::Array{Int64,1}) = ExplainMill.SparseArrayMask(ExplainMill.Mask(m), columns)

@testset "Testing correctness of detecting samples that should not be considered in the calculation of daf values" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(sparse([1 0 3 0 5; 1 2 0 4 0]))

	m = Mask(cn, d -> rand(d))
	invalidate!(m, [1,2])
	@test participate(m) ≈ [false, false, false, true, true, true]
	participate(m) .= true
	invalidate!(m, [2,4])
	@test participate(m) ≈ [true, true, false, true, false, true]

	sn = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
	m = Mask(sn, d -> rand(d))
	invalidate!(m, [2,4])
	@test participate(m) ≈ [true, false, true, false, true]

	on = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
	m = Mask(on, d -> rand(d))
	@test length(prunemask(m.mask)) == 5
	invalidate!(m, [2,4])
	@test participate(m) ≈ [true, false, true, false, true]

	ds = BagNode(sn, AlignedBags([1:2,3:3,4:5]))
	m = Mask(ds, d -> rand(d))
	invalidate!(m, [1,3])
	@test participate(m) ≈ participate(m.child) ≈ [false, false, true, false, false]
	m = Mask(ds, d-> rand(d))
	prunemask(m)[[1,2,4,5]] .= false
	invalidate!(m)
	@test prunemask(m) ≈ [false, false, true, false, false]
	@test all(participate(m) .== true)
	@test participate(m.child) ≈ [false, false, true, false, false]


	ds = BagNode(BagNode(sn, AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))
	m = Mask(ds, d -> rand(d))
	invalidate!(m, [1])
	@test all(prunemask(m) .== true)
	@test all(participate(m) .== false)
	@test all(prunemask(m.child) .== true)
	@test all(participate(m.child) .== false)
	@test all(prunemask(m.child.child) .== true)
	@test all(participate(m.child.child) .== false)
	m = Mask(ds, d -> rand(d))
	prunemask(m)[[1,3]] .= false
	invalidate!(m)
	@test prunemask(m) ≈ [false, true, false]
	@test all(participate(m) .== true)
	@test participate(m.child) ≈ [false, false, true, false, false]
	@test participate(m.child.child) ≈ participate(m.child) ≈ [false, false, true, false, false]

	ds = BagNode(ProductNode((a = cn, b = sn)), AlignedBags([1:2,3:3,4:5]))
	m = Mask(ds, d -> rand(d))
	prunemask(m)[[1,3]] .= false
	invalidate!(m)
	@test prunemask(m) ≈ [false, true, false, true, true]
	@test all(participate(m) .== true)
	@test participate(m.child.childs[:a]) ≈ [false, false, true, false, true, true]
	@test participate(m.child.childs[:b]) ≈ [false, true, false, true, true]
	m = Mask(ds, d -> rand(d))
	invalidate!(m,[1,3])
	@test all(prunemask(m) .== true)
	@test participate(m) ≈ [false, false, true, false, false]
	@test participate(m.child.childs[:a]) ≈ [false, false, false, true, false, false]
	@test participate(m.child.childs[:b]) ≈ [false, false, true, false, false]
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
	@test prunemask(an[:a]) ≈ [true, false, true, true, true]
	@test participate(an[:a]) ≈ [false, true, true, true, true]
	@test prunemask(an[:b]) ≈ [true, false]
	@test participate(an[:b]) ≈ [false, true]
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

@testset "testing multiplication masks" begin 
	#for now I will test that they just pass
	x = ProductNode((a = ArrayNode(randn(Float32, 2,4)), b = ArrayNode(randn(Float32, 2, 4))))
	m = reflectinmodel(x, d -> Dense(d, 4), d -> SegmentedMeanMax(d))
	mk = ExplainMill.Mask(x, m, d -> fill(1f0, d, 1), ExplainMill._nocluster)
	ps = Flux.Params([mk[:a].mask.stats, mk[:b].mask.stats])
	gradient(() -> sum(m(x, mk).data), ps)
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

    # @test_broken begin
    #     # buf = IOBuffer()
    #     # printtree(buf, m, trav=true)
    #     # str_repr = String(take!(buf))
    #     str_repr = ""
    #     str_repr ==
    #     """
    #     BagMask [""]
    #     └── BagMask ["U"]
    #     └── ProductMask ["k"]
    #     ├── a: MatrixMask ["o"]
    #     ├── c: SparseArrayMask ["s"]
    #     └── o: CategoricalMask ["w"]"""
    # end
end
