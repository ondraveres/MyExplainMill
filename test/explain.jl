@testset "test that sampling with clusters as expected" begin
	m = Mask(fill(true, 4), fill(true, 4), fill(1, 4), Daf(2), [1,2,1,2])
	for i in 1:10
		sample!(m)
		@test mask(m)[1] == mask(m)[3] && mask(m)[2] == mask(m)[4]
	end
end

@testset "workflow --- independent instances" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(BagNode(ProductNode((a = an, c = cn)), AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))

	model = reflectinmodel(ds, d -> Dense(d, 1))

	pruning_mask = Mask(ds)
	dafs = []
	mapmask(pruning_mask) do m
		m != nothing && push!(dafs, m)
	end

	sample!(pruning_mask)
	pruned_ds = prune(ds, pruning_mask)
	o = model(pruned_ds)
	Duff.update!(dafs, o, pruning_mask)
	@test true
end

@testset "workflow --- clustered instances" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(BagNode(ProductNode((a = an, c = cn)), AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))

	model = reflectinmodel(ds, d -> Dense(d, 1))

	pruning_mask = Mask(ds, model)
	dafs = []
	mapmask(pruning_mask) do m
		m != nothing && push!(dafs, m)
	end

	sample!(pruning_mask)
	pruned_ds = prune(ds, pruning_mask)
	o = model(pruned_ds).data
	Duff.update!(dafs, o, pruning_mask)
	@test true
end

@testset "getindex / setindex for Mask" begin
	m = Mask(fill(true, 4), fill(true, 4), fill(1, 4), Daf(2), [1,2,1,2])
	m[2] = false
	@test m.mask ≈ [true, false, true, false]
	@test all(m[1] .== true)
	@test all(m[2] .== false)
	m[1] = false
	@test m.mask ≈ [false, false, false, false]

	m = Mask(fill(true, 4), fill(true, 4), fill(1, 4), Daf(2), nothing)
	m[2] = false
	@test m.mask ≈ [true, false, true, true]
	@test m[1] == true
	@test m[2] == false
	m[4] = false
	@test m.mask ≈ [true, false, true, false]
end
