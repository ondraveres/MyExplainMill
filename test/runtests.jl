using ExplainMill
using Mill, SparseArrays
using ExplainMill: prune, Mask, invalidate!, mapmask, participate, mask
using Setfield, JSON
using Test
using MLDataPattern
using StatsBase, Flux, Duff
using Mill: NGramMatrix
using ExplainMill: MatrixMask, TreeMask, BagMask, NGramMatrixMask, SparseArrayMask, CategoricalMask

ExplainMill.Mask(m::Vector{Bool}) = ExplainMill.Mask(m, fill(true, length(m)), fill(0, length(m)), Daf(length(m)), nothing)
ExplainMill.MatrixMask(m::Vector{Bool}) = ExplainMill.MatrixMask(ExplainMill.Mask(m))
ExplainMill.CategoricalMask(m::Vector{Bool}) = ExplainMill.CategoricalMask(ExplainMill.Mask(m))
ExplainMill.BagMask(child, bags, m::Vector{Bool}) = ExplainMill.BagMask(child, bags, ExplainMill.Mask(m))
ExplainMill.NGramMatrixMask(m::Vector{Bool}) = ExplainMill.NGramMatrixMask(ExplainMill.Mask(m))
ExplainMill.SparseArrayMask(m::Vector{Bool}, columns) = ExplainMill.SparseArrayMask(ExplainMill.Mask(m), columns)

@testset "Testing correctness of detecting samples that should not be considered in the calculation of daf values" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(sparse([1 0 3 0 5; 1 2 0 4 0]))

	m = Mask(cn)
	invalidate!(m, [1,2])
	@test participate(m) ≈ [false, false, false, true, true, true]
	participate(m) .= true
	invalidate!(m, [2,4])
	@test participate(m) ≈ [true, true, false, true, false, true]


	sn = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
	m = Mask(sn)
	invalidate!(m, [2,4])
	@test participate(m) ≈ [true, false, true, false, true]

	on = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
	m = Mask(on)
	@test length(mask(m.mask)) == 5
	invalidate!(m, [2,4])
	@test participate(m) ≈ [true, false, true, false, true]

	ds = BagNode(sn, AlignedBags([1:2,3:3,4:5]))
	m = Mask(ds)
	invalidate!(m, [1,3])
	@test participate(m) ≈ participate(m.child) ≈ [false, false, true, false, false]
	m = Mask(ds)
	mask(m)[[1,2,4,5]] .= false
	invalidate!(m)
	@test mask(m) ≈ [false, false, true, false, false]
	@test all(participate(m) .== true)
	@test participate(m.child) ≈ [false, false, true, false, false]


	ds = BagNode(BagNode(sn, AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))
	m = Mask(ds)
	invalidate!(m, [1])
	@test all(mask(m) .== true)
	@test all(participate(m) .== false)
	@test all(mask(m.child) .== true)
	@test all(participate(m.child) .== false)
	@test all(mask(m.child.child) .== true)
	@test all(participate(m.child.child) .== false)
	m = Mask(ds)
	mask(m)[[1,3]] .= false
	invalidate!(m)
	@test mask(m) ≈ [false, true, false]
	@test all(participate(m) .== true)
	@test participate(m.child) ≈ [false, false, true, false, false]
	@test participate(m.child.child) ≈ participate(m.child) ≈ [false, false, true, false, false]

	ds = BagNode(TreeNode((a = cn, b = sn)), AlignedBags([1:2,3:3,4:5]))
	m = Mask(ds)
	mask(m)[[1,3]] .= false
	invalidate!(m)
	@test mask(m) ≈ [false, true, false, true, true]
	@test all(participate(m) .== true)
	@test participate(m.child.childs[:a]) ≈ [false, false, true, false, true, true]
	@test participate(m.child.childs[:b]) ≈ [false, true, false, true, true]
	m = Mask(ds)
	invalidate!(m,[1,3])
	@test all(mask(m) .== true)
	@test participate(m) ≈ [false, false, true, false, false]
	@test participate(m.child.childs[:a]) ≈ [false, false, false, true, false, false]
	@test participate(m.child.childs[:b]) ≈ [false, false, true, false, false]
end


@testset "mapmask" begin
	an = Mask(ArrayNode(reshape(collect(1:10), 2, 5)))
	mapmask(an) do m
		participate(m)[1] = false
		mask(m)[2] = false
	end
	@test mask(an) ≈ [true, false]
	@test participate(an) ≈ [false, true]


	an = Mask(ArrayNode(sparse([1 0 3 0 5; 1 2 0 4 0])))
	mapmask(an) do m
		participate(m)[1] = false
		mask(m)[2] = false
	end
	@test mask(an) ≈ [true, false, true, true, true, true]
	@test participate(an) ≈ [false, true, true, true, true, true]

	an = Mask(ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4)))
	mapmask(an) do m
		participate(m)[1] = false
		mask(m)[2] = false
	end
	@test mask(an) ≈ [true, false, true, true, true]
	@test participate(an) ≈ [false, true, true, true, true]


	an = Mask(ArrayNode(NGramMatrix(["a","b","c"],3,256,2053)))
	mapmask(an) do m
		participate(m)[1] = false
		mask(m)[2] = false
	end
	@test mask(an) ≈ [true, false, true]
	@test participate(an) ≈ [false, true, true]

	an = Mask(TreeNode((a = ArrayNode(NGramMatrix(["a","b","c","d","e"],3,256,2053)),
		b = ArrayNode(reshape(collect(1:10), 2, 5)))))
	mapmask(an) do m
		participate(m)[1] = false
		mask(m)[2] = false
	end
	@test mask(an.childs.a) ≈ [true, false, true, true, true]
	@test participate(an.childs.a) ≈ [false, true, true, true, true]
	@test mask(an.childs.b) ≈ [true, false]
	@test participate(an.childs.b) ≈ [false, true]
	@test participate(an) == nothing
	@test mask(an) == nothing


	an = Mask(BagNode(ArrayNode(reshape(collect(1:10), 2, 5)), AlignedBags([1:2,3:3,4:5])))
	mapmask(an) do m
		participate(m)[1] = false
		mask(m)[2] = false
	end
	@test mask(an) ≈ [true, false, true, true, true,]
	@test participate(an) ≈ [false, true, true, true, true]
	@test mask(an.child) ≈ [true, false]
	@test participate(an.child) ≈ [false, true]
end


@testset "Testing prunning of samples " begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	on = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(BagNode(TreeNode((a = an, c = cn, o = on)), AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))

	m = BagMask(
			BagMask(
				TreeMask((a = MatrixMask([true,false]),
				c = SparseArrayMask([true, true, true, false, true], [1, 2, 3, 4, 5]),
				o = CategoricalMask([true, true, true, false, false]),)
				), ds.bags.bags,
			[true,false,true,false,true]),
			ds.bags,
			[true,true,true])
	dss = prune(ds, m)

	@test nobs(dss) == 1
	@test nobs(dss.data) == 3
	@test nobs(dss.data.data) == 3
	@test dss.data.data.data.a.data ≈ [1 5 9; 0 0 0 ]
	@test dss.data.data.data.c.data.nzval ≈ [1,3,5]
	@test dss.data.data.data.o.data ≈ Flux.onehotbatch([1,3,4], 1:4)

	m = BagMask(
		BagMask(
			TreeMask((a = MatrixMask([false,true]),
			c = SparseArrayMask([false, true, false, true, false], [1, 2, 3, 4, 5]),
			o = CategoricalMask([false, true, false, true, false]),)
			), ds.bags.bags,
		[true,false,true,false,true]),
		ds.bags,
		[true,true,true])
	dss = prune(ds, m)

	@test nobs(dss) == 1
	@test nobs(dss.data) == 3
	@test nobs(dss.data.data) == 3
	@test dss.data.data.data.c.data.nzval ≈ [0, 0, 0]
	@test dss.data.data.data.o.data ≈ Flux.onehotbatch([4,4,4], 1:4)
	@test dss.data.data.data.a.data ≈ [0 0 0; 2 6 10]

	m = BagMask(
		BagMask(
			TreeMask((a = MatrixMask([false,true]),
			c = SparseArrayMask([true, true, true, true, true], [1, 2, 3, 4, 5]),
			o = CategoricalMask([false, true, false, true, false]),)
			), ds.bags.bags,
		[true,false,false,false,true]),
		ds.bags,
		[true,true,true])
	dss = prune(ds, m)

	@test nobs(dss) == 1
	@test nobs(dss.data) == 3
	@test all(dss.data.bags.bags .== [1:1, 0:-1, 2:2])
	@test nobs(dss.data.data) == 2
	@test dss.data.data.data.c.data.nzval ≈ [1, 5]
	@test dss.data.data.data.o.data ≈ Flux.onehotbatch([4,4], 1:4)
	@test dss.data.data.data.a.data ≈ [0 0; 2 10]
end

@testset "testing infering of sample membership" begin
	sn = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
	ds = BagNode(sn, AlignedBags([1:2,3:3,4:5]))
	pm = Mask(ds)
	ExplainMill.infersamplemembership!(pm, nobs(ds))
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
	ds = BagNode(BagNode(TreeNode((a = an, c = cn, o = on)), AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))

	m = ExplainMill.BagMask(
			ExplainMill.BagMask(
				ExplainMill.TreeMask((a = ExplainMill.MatrixMask([true,false]),
				c = ExplainMill.SparseArrayMask([true, true, true, false, true], [1, 2, 3, 4, 5]),
				o = ExplainMill.CategoricalMask([true, true, true, false, false]),)
				), ds.bags.bags,
			[true,false,true,false,true]),
			ds.bags,
			[true,true,true])

		buf = IOBuffer()
		# printtree(buf, m, trav=true)
		Mill.dsprint(buf, m)
		str_repr = String(take!(buf))
		@test str_repr ==
"""
BagMask
  └── BagMask
        └── TreeMask
              ├── a: MatrixMask
              ├── c: SparseArrayMask
              └── o: CategoricalMask"""
end

# @testset "nnodes" begin
#     @test nnodes(ext) == 12
#     @test nnodes(ext[:a]) == 1
#     @test nnodes(ext[:b]) == 4
#     @test nnodes(ext[:c]) == 6
# end
#
# @testset "nleafs" begin
#     @test nleafs(ext[:a]) + nleafs(ext[:b]) + nleafs(ext[:c]) == nleafs(ext)
# end
#
# @testset "children" begin
# 	@test children(ext) == (a=ext[:a], b=ext[:b], c=ext[:c])
# 	@test children(ext[:a]) == []
# 	@test children(ext[:b]) == (a=ext[:b][:a], b=ext[:b][:b])
# 	@test children(ext[:b][:a]) == (ext[:b][:a].item,)
# 	@test children(ext[:b][:b]) == []
# 	@test children(ext[:c]) == (a=ext[:c][:a],)
# 	@test children(ext[:c][:a]) == (a=ext[:c][:a][:a], b=ext[:c][:a][:b])
# 	@test children(ext[:c][:a][:a]) == (ext[:c][:a][:a].item,)
# 	@test children(ext[:c][:a][:b]) == (ext[:c][:a][:b].item,)
# end
#
# @testset "nchildren" begin
# 	@test nchildren(ext) == 3
# 	@test nchildren(ext[:a]) == 0
# 	@test nchildren(ext[:b]) == 2
# 	@test nchildren(ext[:b][:a]) == 1
# 	@test nchildren(ext[:b][:b]) == 0
# 	@test nchildren(ext[:c]) == 1
# 	@test nchildren(ext[:c][:a]) == 2
# 	@test nchildren(ext[:c][:a][:a]) == 1
# 	@test nchildren(ext[:c][:a][:b]) == 1
# end
#
# @testset "getindex on strings" begin
# 	@test ext[""] == ext
# 	@test ext["E"] == ext[:a]
# 	@test ext["U"] == ext[:b]
# 	@test ext["Y"] == ext[:b][:a]
# 	@test ext["a"] == ext[:b][:a].item
# 	@test ext["c"] == ext[:b][:b]
# 	@test ext["k"] == ext[:c]
# 	@test ext["s"] == ext[:c][:a]
# 	@test ext["u"] == ext[:c][:a][:a]
# 	@test ext["v"] == ext[:c][:a][:a].item
# 	@test ext["w"] == ext[:c][:a][:b]
# 	@test ext["x"] == ext[:c][:a][:b].item
# end
