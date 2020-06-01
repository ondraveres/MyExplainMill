using ExplainMill, Mill, SparseArrays, HierarchicalUtils
using ExplainMill: prune, Mask, invalidate!, mapmask, participate, mask
using ExplainMill: ProductMask, BagMask, MatrixMask, EmptyMask, SparseArrayMask, identifycolumns, CategoricalMask, NGramMatrixMask
using Test
using MLDataPattern
using StatsBase, Flux, Duff


function mulnnz(x::SparseMatrixCSC, mask)
end

@testset "do we handle correctly the explanation?" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(Flux.onehotbatch([1,2,3,2,1], 1:3))
	sn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	zn = ArrayNode(Mill.NGramMatrix(["a","b","c","d","e"], 3, 256, 2053))
	ds = BagNode(BagNode(ProductNode((
			a = an, 
			c = cn, 
			# s = sn, 
			z = zn)), 
		AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))

	model = reflectinmodel(ds, d -> Dense(d, 3), d -> SegmentedMean(d))


	m = BagMask(BagMask(ProductMask((
		a = MatrixMask(rand(2), 2, 1), 
		c = CategoricalMask(rand(5)),
		# s = SparseArrayMask(rand(nnz(sn.data)), identifycolumns(sn.data)), 
		z = NGramMatrixMask(rand(5)),
		)), 
		ds.data.bags, rand(5)), ds.data.bags, rand(3))
	model(ds, m)

	mo₁, ma₁, d₁ = model.im.im[:a], m.child.child[:a], ds.data.data[:a]
	@test mo₁(d₁, ma₁).data ≈ mo₁(ArrayNode(d₁.data .* ma₁.mask)).data
	gs = gradient(() -> sum(mo₁(d₁, ma₁).data), Flux.Params([ma₁.mask]))
	@test !isnothing(gs[ma₁.mask])

	mo₁, ma₁, d₁ = model.im.im[:c], m.child.child[:c], ds.data.data[:c]
	@test mo₁(d₁, ma₁).data ≈ mo₁(ArrayNode(d₁.data .* transpose(ma₁.mask))).data
	gs = gradient(() -> sum(mo₁(d₁, ma₁).data), Flux.Params([ma₁.mask]))
	@test !isnothing(gs[ma₁.mask])

	# mo₁, ma₁, d₁ = model.im.im[:s], m.child.child[:s], ds.data.data[:s]
	# gs = gradient(() -> sum(mo₁(d₁, ma₁)), Flux.Params([ma₁.mask]))
	# @test !isnothing(gs[ma₁.mask])

	mo₁, ma₁, d₁ = model.im.im[:z], m.child.child[:z], ds.data.data[:z]
	@test mo₁(d₁, ma₁).data ≈ mo₁(ArrayNode(SparseMatrixCSC(d₁.data) .* transpose(ma₁.mask))).data
	gs = gradient(() -> sum(mo₁(d₁, ma₁).data), Flux.Params([ma₁.mask]))
	@test !isnothing(gs[ma₁.mask])

	ms = filter(x -> !isa(x,ExplainMill.AbstractNoMask), collect(NodeIterator(m)))
	ps = Flux.Params(map(x -> x.mask, ms))
	gs = gradient(() -> sum(model(ds,m).data), ps)
	@test all([!isnothing(gs[p]) for p in ps])
end

