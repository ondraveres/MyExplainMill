using ExplainMill
using Mill
using Test
using Flux
using SparseArrays
using Setfield
using ExplainMill: SimpleMask, create_mask_structure, foreach_mask, collect_masks_with_levels, mapmask
using ExplainMill: CategoricalMask, MatrixMask, NGramMatrixMask, SparseArrayMask, BagMask, ProductMask, FollowingMasks
using ExplainMill: ParticipationTracker, participate, invalidate!, present
using FiniteDifferences
using StatsBase: nobs

function testmaskgrad(f, ps::Flux.Params; detailed = false, ϵ = 1e-6)
    gs = gradient(f, ps)
    o = map(p -> graddifference(fdmgradient(f, p), gs[p]), ps)
    detailed ? o : all(o .< ϵ)
end

graddifference(x::AbstractArray, y::AbstractArray) = maximum(abs.(x .- y))
graddifference(x::AbstractArray, y::Nothing) = maximum(abs.(x))

function fdmgradient(f, p)
    op = deepcopy(p)
    function fp(x)
        p .= x 
        f()
    end
    fval = grad(central_fdm(5, 1), fp, p)[1]
    p .= op 
    fval
end

@testset "Structural masks" begin
	@testset "MatrixMask" begin
		an = ArrayNode(randn(4,5))
		mk = create_mask_structure(an, d -> SimpleMask(fill(true, d)))
		@test mk isa MatrixMask
		@test an[mk] == an

		#this is not really a nice syntax
		mk.mask.x[[1,3]] .= false
		@test mk.mask.x == [false, true, false, true]

		# testing basic subsetting of data 
		@test an[mk].data ≈ an.data .* [false, true, false, true]	

		#test indication of presence of observations
		x = deepcopy(mk.mask.x)
		@test present(mk, [true, false, true, false, true]) == [true, false, true, false, true]
		mk.mask.x .= false
		@test present(mk, [true, false, true, false, true]) == fill(false, 5)
		mk.mask.x .= x

		# testing subsetting while exporting only subset of observations
		@test an[mk, [true, false, true, false, true]].data ≈ an.data[:,[true, false, true, false, true]] .* [false, true, false, true]	

		# multiplication is equicalent to subsetting
		model = f64(reflectinmodel(an, d -> Dense(d, 10)))
		@test model(an[mk]).data ≈ model(an, mk).data


		# calculation of gradient with respect to boolean mask
		gs = gradient(() -> sum(model(an, mk).data),  Flux.Params([mk.mask.x]))
		@test all(abs.(gs[mk.mask.x]) .> 0)

		# Verify that calculation of the gradient for real mask is correct 
		mk = create_mask_structure(an, d -> SimpleMask(rand(d)))
		ps = Flux.Params([mk.mask.x])
		testmaskgrad(() -> sum(model(an, mk).data),  ps)

		# testing foreach_mask
		cmk = collect_masks_with_levels(mk; level = 2)
		@test length(cmk) == 1
		@test cmk[1].first == mk.mask
		@test cmk[1].second == 2

		# testing mapmask
		cmk = mapmask(mk) do m, l
			SimpleMask(m.x .+ 1)
		end
		@test cmk.mask.x ≈ mk.mask.x .+ 1
		# update of the participation (does not apply now)

		#test sub-indexing with the structural mask 
		mk = ObservationMask(SimpleMask(fill(true, nobs(an))))
		@test an[mk] == an
		mk.mask.x[1] = false
		@test an[mk].data == an.data[:,2:end]
		@test model(an, mk).data ≈ model(ArrayNode(an.data .* [0 1 1 1 1])).data
	end

	@testset "Categorical Mask" begin
		on = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
		mk₁ = create_mask_structure(on, d -> SimpleMask(fill(true, d)))
		@test mk₁ isa CategoricalMask
		mk₂ = ObservationMask(SimpleMask(fill(true, nobs(on))))
		for mk in [mk₁, mk₂]
			@test on[mk] == on

			# subsetting
			mk.mask.x[[1,3]] .= false
			@test mk.mask.x == [false, true, false, true, true]

			# test pruning of samples 
			@test on[mk].data ≈ Flux.onehotbatch([4, 2, 4, 1, 2], 1:4)

			#test indication of presence of observations
			@test present(mk, [true, false, true, false, true]) == [false, false, false, false, true]

			# testing subsetting while exporting only subset of observations
			@test on[mk, [true, false, true, false, true]].data ≈ Flux.onehotbatch([4, 4, 2], 1:4)

			# output of a model on pruned sample is equal to output multiplicative weights
			model = f64(reflectinmodel(on, d -> Chain(Dense(d, 10), Dense(10,10))))
			@test model(on[mk]).data ≈ model(on, mk).data

			# calculation of gradient with respect to boolean mask
			gs = gradient(() -> sum(model(on, mk).data),  Flux.Params([mk.mask.x]))
			@test all(abs.(gs[mk.mask.x]) .> 0)

			# Verify that calculation of the gradient for real mask is correct 
			mk = create_mask_structure(on, d -> SimpleMask(rand(d)))
			ps = Flux.Params([mk.mask.x])
			testmaskgrad(() -> sum(model(on, mk).data),  ps)

			# testing foreach_mask
			cmk = collect_masks_with_levels(mk; level = 2)
			@test length(cmk) == 1
			@test cmk[1].first == mk.mask
			@test cmk[1].second == 2

			# testing mapmask
			cmk = mapmask(mk) do m, l
				SimpleMask(m.x .+ 1)
			end
			@test cmk.mask.x ≈ mk.mask.x .+ 1
		end

		# update of the participation
		on = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
		mk₁ = create_mask_structure(on, d -> ParticipationTracker(SimpleMask(fill(true, d))))
		@test mk₁ isa CategoricalMask
		mk₂ = ObservationMask(ParticipationTracker(SimpleMask(fill(true, nobs(on)))))
		for mk in [mk₁, mk₂]	
			@test all(participate(mk.mask))
			invalidate!(mk, [1])
			@test participate(mk.mask) == Bool[0, 1, 1, 1, 1]
			invalidate!(mk, [1, 3])
			@test participate(mk.mask) == Bool[0, 1, 0, 1, 1]
			ExplainMill.updateparticipation!(mk)
			@test all(participate(mk.mask))
		end
	end

	@testset "Sparse Mask" begin
		cn = ArrayNode(sparse(Float64.([1 2 3 0 5; 0 2 0 4 0])))
		mk = create_mask_structure(cn, d -> SimpleMask(fill(true, d)))
		@test mk isa SparseArrayMask
		@test cn[mk] == cn

		# subsetting
		mk.mask.x[[1,3]] .= false
		@test mk.mask.x == [false, true, false, true, true, true]

		# testing subsetting while exporting only subset of observations
		@test cn[mk, [true, false, true, false, true]].data ≈ sparse(Float64.([0 3 5; 0 0 0]))

		#test indication of presence of observations
		x = deepcopy(mk.mask.x)
		@test present(mk, [true, true, true, false, false]) == [false, true, true, false, false]
		mk.mask.x[2:3] .= false
		@test present(mk, [true, true, true, false, false]) == [false, false, true, false, false]
		@test present(mk, [true, true, true, false, true]) == [false, false, true, false, true]
		mk.mask.x .= x

		# multiplication is equicalent to subsetting
		@test cn[mk].data.nzval == [0, 2, 0, 3, 4, 5]

		# calculation of gradient with respect to boolean mask
		model = f64(reflectinmodel(cn, d -> Chain(Dense(d, 10), Dense(10,10))))
		@test model(cn[mk]).data ≈ model(cn, mk).data

		# Verify that calculation of the gradient for real mask is correct 
		gs = gradient(() -> sum(model(cn, mk).data),  Flux.Params([mk.mask.x]))
		@test all(abs.(gs[mk.mask.x]) .> 0)

		mk = create_mask_structure(cn, d -> SimpleMask(rand(d)))
		ps = Flux.Params([mk.mask.x])
		testmaskgrad(() -> sum(model(cn, mk).data),  ps)

		# testing foreach_mask
		cmk = collect_masks_with_levels(mk; level = 2)
		@test length(cmk) == 1
		@test cmk[1].first == mk.mask
		@test cmk[1].second == 2

		# testing mapmask
		cmk = mapmask(mk) do m, l
			SimpleMask(m.x .+ 1)
		end
		@test cmk.mask.x ≈ mk.mask.x .+ 1


		# update of the participation
		cn = ArrayNode(sparse(Float64.([1 2 3 0 5; 0 2 0 4 0])))
		mk = create_mask_structure(cn, d -> ParticipationTracker(SimpleMask(fill(true, d))))
		
		@test all(participate(mk.mask))
		invalidate!(mk, [1])
		@test participate(mk.mask) == Bool[0, 1, 1, 1, 1, 1]
		invalidate!(mk, [1, 2])
		@test participate(mk.mask) == Bool[0, 0, 0, 1, 1, 1]
		ExplainMill.updateparticipation!(mk)
		@test all(participate(mk.mask))

		#test sub-indexing with the structural mask 
		mk = ObservationMask(SimpleMask(fill(true, nobs(cn))))
		@test cn[mk] == cn
		mk.mask.x[2] = false
		@test cn[mk].data == [1 0 3 0 5; 0 0 0 4 0]
		@test model(cn, mk).data ≈ model(ArrayNode(cn.data .* [1 0 1 1 1])).data
	end

	@testset "String Mask" begin
		sn = ArrayNode(NGramMatrix(string.([1,2,3,4,5]), 3, 256, 2053))
		mk₁ = create_mask_structure(sn, d -> SimpleMask(fill(true, d)))
		@test mk₁ isa NGramMatrixMask

		mk₂ = ObservationMask(SimpleMask(fill(true, nobs(sn))))
		for mk in [mk₁, mk₂]
			@test sn[mk] == sn

			# subsetting
			mk.mask.x[[1,3]] .= false
			@test mk.mask.x == [false, true, false, true, true]
			@test sn[mk].data == NGramMatrix(["", "2", "", "4", "5"], 3, 256, 2053)

			# testing subsetting while exporting only subset of observations
			@test sn[mk, [true, false, true, false, true]].data == NGramMatrix(["", "", "5"], 3, 256, 2053)

			#test indication of presence of observations
			@test present(mk, [true, false, true, false, true]) == [false, false, false, false, true]

			# multiplication is equicalent to subsetting
			model = f64(reflectinmodel(sn, d -> Chain(Dense(d, 10), Dense(10,10))))
			@test model(sn[mk]).data ≈ model(sn, mk).data

			# Verify that calculation of the gradient for real mask is correct 
			gs = gradient(() -> sum(model(sn, mk).data),  Flux.Params([mk.mask.x]))
			@test all(abs.(gs[mk.mask.x]) .> 0)

			mk = create_mask_structure(sn, d -> SimpleMask(rand(d)))
			ps = Flux.Params([mk.mask.x])
			testmaskgrad(() -> sum(model(sn, mk).data),  ps)

			# testing foreach_mask
			cmk = collect_masks_with_levels(mk; level = 2)
			@test length(cmk) == 1
			@test cmk[1].first == mk.mask
			@test cmk[1].second == 2

			# testing mapmask
			cmk = mapmask(mk) do m, l
				SimpleMask(m.x .+ 1)
			end
			@test cmk.mask.x ≈ mk.mask.x .+ 1
		end

		# update of the participation (does not apply now)
		sn = ArrayNode(NGramMatrix(string.([1,2,3,4,5]), 3, 256, 2053))
		mk = create_mask_structure(sn, d -> ParticipationTracker(SimpleMask(fill(true, d))))
		
		@test all(participate(mk.mask))
		invalidate!(mk, [1])
		@test participate(mk.mask) == Bool[0, 1, 1, 1, 1]
		invalidate!(mk, [1, 3])
		@test participate(mk.mask) == Bool[0, 1, 0, 1, 1]
		ExplainMill.updateparticipation!(mk)
		@test all(participate(mk.mask))
	end

	@testset "Bag Mask --- single nesting" begin
		an = ArrayNode(randn(4,5))
		ds = BagNode(an, AlignedBags([1:2,3:3,0:-1,4:5]))
		mk = create_mask_structure(ds, d -> SimpleMask(fill(true, d)))
		@test mk isa BagMask
		@test ds[mk] == ds

		# subsetting
		mk.mask.x[[1,3]] .= false
		@test mk.mask.x == [false, true, false, true, true]

		@test ds[mk].data == ds.data[[false, true, false, true, true]]
		@test ds[mk].bags.bags ==  UnitRange{Int64}[1:1, 0:-1, 0:-1, 2:3]

		#test indication of presence of observations
		x = deepcopy(mk.child.mask.x)
		@test present(mk, [true, false, true, true]) == [true, false, false, true]
		mk.child.mask.x .= false
		@test present(mk, [true, false, true, true]) == [false, false, false, false]
		mk.child.mask.x .= x

		# If child exports nothing, we return only empty bags
		x = deepcopy(mk.child.mask.x)
		mk.child.mask.x .= false
		@test ds[mk].bags.bags ==  UnitRange{Int64}[0:-1, 0:-1, 0:-1, 0:-1]
		@test isempty(ds[mk].data.data)
		mk.child.mask.x .= x

		# testing subsetting while exporting only subset of observations
		@test ds[mk, [true, true, false, false]].data == ds.data[[false, true, false, false, false]]
		@test ds[mk, [true, true, false, false]].bags.bags ==  UnitRange{Int64}[1:1, 0:-1]

		# prepare there models for the test
		model₁ = f64(reflectinmodel(ds, d -> Chain(Dense(d, 10), Dense(10,10)), Mill.SegmentedMax))
		model₁.a.C .= minimum(model₁.im(ds.data).data, dims = 2)[:]

		model₂ = f64(reflectinmodel(ds, d -> Chain(Dense(d, 10), Dense(10,10)), Mill.SegmentedMean))
		model₂.a.C .= 0

		model₃ = f64(reflectinmodel(ds, d -> Chain(Dense(d, 10), Dense(10,10)), Mill.SegmentedMeanMax))
		model₃.a[1].C .= 0
		model₃.a[2].C .= minimum(model₃.im(ds.data).data, dims = 2)[:]

		for model in [model₁, model₂, model₃]
			# multiplication is equivalent to subsetting
			mk = create_mask_structure(ds, d -> SimpleMask(fill(true, d)))
			mk.mask.x[[1,3]] .= false
			@test model(ds[mk]).data ≈ model(ds, mk).data

			# If child exports nothing, we return only empty bags
			x = deepcopy(mk.child.mask.x)
			mk.child.mask.x .= false
			@test model(ds[mk]).data ≈ model(ds, mk).data
			mk.child.mask.x .= x

			# Verify that calculation of the gradient for real mask is correct 
			gs = gradient(() -> sum(model(ds, mk).data),  Flux.Params([mk.mask.x]))
			@test sum(abs.(gs[mk.mask.x])) > 0

			mk = create_mask_structure(ds, d -> SimpleMask(rand(d)))
			ps = Flux.Params([mk.mask.x])
			testmaskgrad(() -> sum(model(ds, mk).data),  ps)
		end

		# testing foreach_mask
		cmk = collect_masks_with_levels(mk; level = 2)
		@test length(cmk) == 2
		@test cmk[1].first == mk.mask
		@test cmk[1].second == 2
		@test cmk[2].first == mk.child.mask
		@test cmk[2].second == 3

		# testing mapmask
		cmk = mapmask(mk) do m, l
			SimpleMask(m.x .+ 1)
		end
		@test cmk.mask.x ≈ mk.mask.x .+ 1
		@test cmk.child.mask.x ≈ mk.child.mask.x .+ 1

		cmk = mapmask(mk) do m, l
			if l == 1
				SimpleMask(m.x .+ 1)
			else 
				m
			end
		end
		@test cmk.mask.x ≈ mk.mask.x .+ 1
		@test cmk.child.mask.x ≈ mk.child.mask.x

		# update of the participation (does not apply now)
		cn = ArrayNode(sparse(Float64.([1 2 3 0 5; 0 2 0 4 0])))
		ds = BagNode(cn, AlignedBags([1:2,3:3,0:-1,4:5]))
		mk = create_mask_structure(ds, d -> ParticipationTracker(SimpleMask(fill(true, d))))
		
		@test all(participate(mk.mask))
		@test all(participate(mk.child.mask))
		ExplainMill.updateparticipation!(mk)
		invalidate!(mk, [1])
		@test participate(mk.mask) == Bool[0, 0, 1, 1, 1]
		@test participate(mk.child.mask) == Bool[0, 0, 0, 1, 1, 1]
		invalidate!(mk, [1, 3])
		@test participate(mk.mask) == Bool[0, 0, 1, 1, 1]
		@test participate(mk.child.mask) == Bool[0, 0, 0, 1, 1, 1]

		ExplainMill.updateparticipation!(mk)
		@test all(participate(mk.mask))
		@test all(participate(mk.child.mask))

		# test the update of participation correctly propagates to childs
		mk.mask[1] = false
		ExplainMill.updateparticipation!(mk)
		@test all(participate(mk.mask))
		@test participate(mk.child.mask) == Bool[0, 1, 1, 1, 1, 1]
	end

	@testset "ProductMask" begin
		ds = ProductNode(
			(a = ArrayNode(randn(4,5)),
			b = ArrayNode(sparse(Float64.([1 2 3 0 5; 0 2 0 4 0]))),
			))
		mk = create_mask_structure(ds, d -> SimpleMask(fill(true, d)))
		@test mk isa ProductMask
		@test ds[mk] == ds

		#this is not really a nice syntax
		mk[:a].mask.x[1:2] .= false
		mk[:b].mask.x[1:2] .= false

		@test ds[mk][:a].data ≈ ds[:a].data .* [false, false, true, true]	
		@test ds[mk][:b].data ≈ sparse(Float64[0 0 3 0 5; 0 2 0 4 0])

		@test ds[mk, [true, false, true, false, true]][:a].data ≈ ds[:a].data[:,[true, false, true, false, true]] .* [false, false, true, true]	
		@test ds[mk, [true, false, true, false, true]][:b].data ≈ sparse(Float64[0 3 5; 0 0 0])

		#test indication of presence of observations
		xa = deepcopy(mk[:a].mask.x)
		xb = deepcopy(mk[:b].mask.x)
		@test present(mk, [true, true, true, false, true]) == [true, true, true, false, true]
		mk[:a].mask.x .= false
		@test present(mk, [true, true, true, false, true]) == [false, true, true, false, true]
		mk[:b].mask.x[3] = false
		@test present(mk, [true, true, true, false, true]) == [false, false, true, false, true]
		mk[:a].mask.x .= xa
		mk[:b].mask.x .= xb

		# multiplication is equicalent to subsetting
		model = f64(reflectinmodel(ds, d -> Dense(d, 10)))
		@test model(ds[mk]).data ≈ model(ds, mk).data

		# calculation of gradient with respect to boolean mask
		gs = gradient(() -> sum(model(ds, mk).data),  Flux.Params([mk[:a].mask.x]))
		@test all(abs.(gs[mk[:a].mask.x]) .> 0)
		gs = gradient(() -> sum(model(ds, mk).data),  Flux.Params([mk[:b].mask.x]))
		@test all(abs.(gs[mk[:b].mask.x]) .> 0)

		# Verify that calculation of the gradient for real mask is correct 
		mk = create_mask_structure(ds, d -> SimpleMask(rand(d)))
		testmaskgrad(() -> sum(model(ds, mk).data), Flux.Params([mk[:a].mask.x]))
		testmaskgrad(() -> sum(model(ds, mk).data), Flux.Params([mk[:b].mask.x]))

		# testing foreach_mask
		cmk = collect_masks_with_levels(mk; level = 2)
		@test length(cmk) == 2
		@test cmk[1].first == mk[:a].mask
		@test cmk[1].second == 2
		@test cmk[2].first == mk[:b].mask
		@test cmk[2].second == 2

		cmk = mapmask(mk) do m, l
			SimpleMask(m.x .+ 1)
		end
		@test cmk[:a].mask.x ≈ mk[:a].mask.x .+ 1
		@test cmk[:b].mask.x ≈ mk[:b].mask.x .+ 1

		# update of the participation
		mk = create_mask_structure(ds, d -> ParticipationTracker(SimpleMask(fill(true, d))))
		@test all(participate(mk[:a].mask))
		@test all(participate(mk[:b].mask))
		invalidate!(mk, [1])
		@test participate(mk[:a].mask) == Bool[1, 1, 1, 1]
		@test participate(mk[:b].mask) == Bool[0, 1, 1, 1, 1, 1]
		invalidate!(mk, [1, 2])
		@test participate(mk[:a].mask) == Bool[1, 1, 1, 1]
		@test participate(mk[:b].mask) == Bool[0, 0, 0, 1, 1, 1]

		ExplainMill.updateparticipation!(mk)
		@test all(participate(mk[:a].mask))
		@test all(participate(mk[:b].mask))
	end

	@testset "simple sharing of masks" begin 
		ds = specimen_sample()
		mk = create_mask_structure(ds, d -> SimpleMask(fill(true, d)))

		# test that mapping of mask works as intended
		mk = mapmask((m, l) -> ParticipationTracker(m), mk)
		cmk = collect_masks_with_levels(mk)
		@test length(cmk) == 6

		mk = create_mask_structure(ds, d -> SimpleMask(fill(true, d)))
		shared_m = ObservationMask(SimpleMask(trues(5)))
		@set! mk.child.child.childs.an = shared_m
		@set! mk.child.child.childs.cn = shared_m
		@set! mk.child.child.childs.on = shared_m
		@set! mk.child.child.childs.sn = shared_m

		@test ds[mk] == ds

		# test synchronous subsetting
		mk.child.child[:an].mask.x[1] = false
		@test ds[mk].data.data[:an].data == ds.data.data[:an].data[:,2:end]
		@test ds[mk].data.data[:cn].data == ds.data.data[:cn].data[:,2:end]
		@test ds[mk].data.data[:sn].data == ds.data.data[:sn].data[2:end]
		@test ds[mk].data.data[:on].data == ds.data.data[:on].data[:,2:end]

		#test that mapping work as intended and preserves the sharing
		mk = mapmask((m, l) -> ParticipationTracker(m), mk)
		cmk = collect_masks_with_levels(mk)
		@test length(cmk) == 3
		@test mk.child.child[:an].mask isa ParticipationTracker
		@test mk.child.child[:cn].mask === mk.child.child[:an].mask
		@test mk.child.child[:sn].mask === mk.child.child[:an].mask
		@test mk.child.child[:on].mask === mk.child.child[:an].mask

		#we should check the gradient and also if model(ds, mk) == model(ds[mk])
	end

	@testset "Leader / follower sharing of masks" begin 
		ds = specimen_sample()
		mk = create_mask_structure(ds, d -> SimpleMask(fill(true, d)))

		shared_m = ObservationMask(SimpleMask(trues(5)))
		#sn is going to be a follower, because it is easy to check
		@set! mk.child.child.childs.an.mask = ObservationMask(SimpleMask(trues(5)))
		@set! mk.child.child.childs.cn.mask = ObservationMask(SimpleMask(trues(5)))
		@set! mk.child.child.childs.on.mask = ObservationMask(SimpleMask(trues(5)))
		@set! mk.child.child.childs.sn = ObservationMask(FollowingMasks((
			mk.child.child.childs.an.mask,
			mk.child.child.childs.cn.mask,
			mk.child.child.childs.on.mask,
			)))

		@test ds[mk] == ds

		# test if everything is unset and we set one of the leaders, followers will be set as well

		# check the mapmask

		# check the mapmask with a different combination of leader / follower to see 
		# that mapmask is permutation invariant

		#we should check the gradient and also if model(ds, mk) == model(ds[mk])


	end
end

@testset "An integration test of nested samples" begin
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	on = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
	cn = ArrayNode(sparse([1 0 3 0 5; 0 2 0 4 0]))
	ds = BagNode(BagNode(ProductNode((a = an, c = cn, o = on)), AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))

	mk = create_mask_structure(ds, d -> SimpleMask(fill(true, d)))
	mk.child.child[:a].mask.x[2] = false
	mk.child.child[:c].mask.x .= [true, true, true, false, true]
	mk.child.child[:o].mask.x .= [true, true, true, false, false]
	mk.child.mask.x .= [true,false,true,false,true]
	dss = ds[mk]

	@test nobs(dss) == 1
	@test nobs(dss.data) == 3
	@test nobs(dss.data.data) == 3
	@test dss.data.data.data.a.data ≈ [1 5 9; 0 0 0 ]
	@test dss.data.data.data.c.data.nzval ≈ [1,3,5]
	@test dss.data.data.data.o.data ≈ Flux.onehotbatch([1,3,4], 1:4)

	mk.child.child[:a].mask.x .= [false, true]
	mk.child.child[:c].mask.x .= [false, true, false, true, false]
	mk.child.child[:o].mask.x .= [false, true, false, true, false]
	mk.child.mask.x .= [true,false,true,false,true]
	dss = ds[mk]

	@test nobs(dss) == 1
	@test nobs(dss.data) == 3
	@test nobs(dss.data.data) == 3
	@test dss.data.data.data.c.data.nzval ≈ [0, 0, 0]
	@test dss.data.data.data.o.data ≈ Flux.onehotbatch([4,4,4], 1:4)
	@test dss.data.data.data.a.data ≈ [0 0 0; 2 6 10]

	mk.child.child[:a].mask.x .= [false, true]
	mk.child.child[:c].mask.x .= [true, true, true, true, true]
	mk.child.child[:o].mask.x .= [false, true, false, true, false]
	mk.child.mask.x .= [true, false, false, false, true]
	dss = ds[mk]

	@test nobs(dss) == 1
	@test nobs(dss.data) == 2
	@test all(dss.data.bags.bags .== [1:1, 2:2])
	@test nobs(dss.data.data) == 2
	@test dss.data.data.data.c.data.nzval ≈ [1, 5]
	@test dss.data.data.data.o.data ≈ Flux.onehotbatch([4,4], 1:4)
	@test dss.data.data.data.a.data ≈ [0 0; 2 10]

	@test ds[ExplainMill.EmptyMask()] == ds
end