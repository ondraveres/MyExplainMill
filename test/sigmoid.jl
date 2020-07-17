@testset "rescaling on basic layers" begin 
	x = randn(3,100)
	c_min, c_max = minimax(x)
	n_min, n_max = [-1,-1,-1], [2,2,2]
	ds = Dense(rescale(c_min, c_max, n_min, n_max)..., identity)
	mn, mx = minimax(ds(x))
	@test isapprox(mx - mn, [3,3, 3], atol = 1e-5)
	@test isapprox(mn, n_min, atol = 1e-5)
	@test isapprox(mx, n_max, atol = 1e-5)

	
	d = Dense(randn(2,3), randn(2), identity)
	xref = d(ds(x))
	fuseaffine!(d, ds)
	@test d(x) ≈ xref
end

@testset "sigmoid on basic layers" begin 
	x = randn(3, 100);
	d = Dense(3,2,relu);

	ds = sigmoid(d, 0.1, x)
	maximum(ds(x), dims = 2)
	minimum(ds(x), dims = 2)
	@test all(maximum(ds(x), dims = 2) .< 0.91)
	@test all(minimum(ds(x), dims = 2) .< 0.01)

	d = Dense(3,2);
	ds = sigmoid(d, 0.1, x)
	@test all(maximum(ds(x), dims = 2) .< 0.91)
	@test all(minimum(ds(x), dims = 2) .> 0.09)

	d = Chain(Dense(3,2,relu), Dense(2,2))
	ds = sigmoid(d, 0.1, x)
	@test all(maximum(ds(x), dims = 2) .< 0.91)
	@test all(minimum(ds(x), dims = 2) .> 0.09)	
end



@testset "sigmoid mill structures" begin 
	bags = Mill.length2bags(rand(1:10, 100))
	xx = randn(3, bags.bags[end].stop);
	@testset "ArrayModel" begin 
		x = ArrayNode(xx)
		m = reflectinmodel(x, d -> Dense(d,2,relu), d -> SegmentedMean(d))

		ms = sigmoid(m, 0.1, x)
		maximum(ms(x).data, dims = 2)
		minimum(ms(x).data, dims = 2)
		@test all(maximum(ms(x).data, dims = 2)  .< 0.91)
		@test all(minimum(ms(x).data, dims = 2)  .< 0.01)
	end


	@testset "BagModel" begin 
		x = BagNode(ArrayNode(xx), bags)
		m = reflectinmodel(x, d -> Dense(d,2,relu), d -> SegmentedMean(d))
		ms = sigmoid(deepcopy(m), 0.1, x)
		@test all(maximum(ms.im(x.data).data, dims = 2)  .< 0.91)
		@test all(minimum(ms.im(x.data).data, dims = 2)  .< 0.01)
		maximum(ms(x).data, dims = 2)
		maximum(m(x).data, dims = 2)
	end

	@testset "ProductModel" begin
		x = ProductNode((
			a = BagNode(ArrayNode(xx), bags),
			b = ArrayNode(randn(2,100))
			))
		m = reflectinmodel(x, d -> Dense(d,2,relu), d -> SegmentedMean(d))
		ms = sigmoid(deepcopy(m), 0.1, x)
		maximum(ms(x).data, dims = 2)
		maximum(m(x).data, dims = 2)
	end
end


