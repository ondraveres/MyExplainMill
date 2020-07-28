function mulnnz(x::SparseMatrixCSC, mask)
end

initmask(d) = ExplainMill.Mask(d, d -> rand(d))

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
		a = MatrixMask(initmask(2), 2, 1), 
		c = CategoricalMask(initmask(5)),
		# s = SparseArrayMask(initmask(nnz(sn.data)), identifycolumns(sn.data)), 
		z = NGramMatrixMask(initmask(5)),
		)), 
		ds.data.bags, initmask(5)), ds.data.bags, initmask(3))
	model(ds, m)

	mo₁, ma₁, d₁ = model.im.im[:a], m.child.child[:a], ds.data.data[:a]
	@test mo₁(d₁, ma₁).data ≈ mo₁(ArrayNode(d₁.data .* σ.(ma₁.mask.stats))).data
	gs = gradient(() -> sum(mo₁(d₁, ma₁).data), Flux.Params([ma₁.mask.stats]))
	@test !isnothing(gs[ma₁.mask.stats])

	mo₁, ma₁, d₁ = model.im.im[:c], m.child.child[:c], ds.data.data[:c]
	@test mo₁(d₁, ma₁).data ≈ mo₁(ArrayNode(d₁.data .* transpose(σ.(ma₁.mask.stats)))).data
	gs = gradient(() -> sum(mo₁(d₁, ma₁).data), Flux.Params([ma₁.mask.stats]))
	@test !isnothing(gs[ma₁.mask.stats])

	# mo₁, ma₁, d₁ = model.im.im[:s], m.child.child[:s], ds.data.data[:s]
	# gs = gradient(() -> sum(mo₁(d₁, ma₁)), Flux.Params([ma₁.mask]))
	# @test !isnothing(gs[ma₁.mask])

	mo₁, ma₁, d₁ = model.im.im[:z], m.child.child[:z], ds.data.data[:z]
	@test mo₁(d₁, ma₁).data ≈ mo₁(ArrayNode(SparseMatrixCSC(d₁.data) .* transpose(σ.(ma₁.mask.stats)))).data
	gs = gradient(() -> sum(mo₁(d₁, ma₁).data), Flux.Params([ma₁.mask.stats]))
	@test !isnothing(gs[ma₁.mask.stats])

	ms = filter(x -> !isa(x,ExplainMill.AbstractNoMask), collect(NodeIterator(m)))
	ps = Flux.Params(map(x -> x.mask.stats, ms))
	gs = gradient(() -> sum(model(ds,m).data), ps)
	@test all([!isnothing(gs[p]) for p in ps])

	ds = BagNode(BagNode(ProductNode((an, cn, zn)), 
		AlignedBags([1:2,3:3,4:5])), 
	AlignedBags([1:3]))
	model = reflectinmodel(ds, d -> Dense(d, 3), d -> SegmentedMean(d))
	m = BagMask(BagMask(ProductMask((
		MatrixMask(initmask(2), 2, 1), 
		CategoricalMask(initmask(5)),
		NGramMatrixMask(initmask(5)),
		)), 
		ds.data.bags, initmask(5)), ds.data.bags, initmask(3))
	model(ds, m)

	ms = filter(x -> !isa(x,ExplainMill.AbstractNoMask), collect(NodeIterator(m)))
	ps = Flux.Params(map(x -> x.mask.stats, ms))
	gs = gradient(() -> sum(model(ds,m).data), ps)
	@test all([!isnothing(gs[p]) for p in ps])
end

