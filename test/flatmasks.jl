@testset "mapping between flat structure and nodes" begin
	for mfun in [
		d -> SimpleMask(fill(true, d)),
		d -> HeuristicMask(ones(Float32, d)),
		d -> ParticipationTracker(SimpleMask(fill(true, d))),
		d -> ParticipationTracker(HeuristicMask(ones(Float32, d))),
		]

		ds = specimen_sample()
		mk = create_mask_structure(ds, mfun)
		fv = FlatView(mk)
		test_participation =  hasmethod(participate, Tuple{typeof(mfun(1))})

		@test length(fv) == 25

		@testset "Testing setindex / getindex" begin
			for i in 1:3
				fv[i] = false
				@test fv[i] == false
				@test mk.mask[i] == false

				fv[i] = true
				@test fv[i] == true
				@test mk.mask[i] == true

				if test_participation
					# participate(mk.mask) is a bit shady, as it can return a copy, but I hope it will work in general
					participate(mk.mask)[i] = false
					@test participate(fv)[i] == false
					participate(mk.mask)[i] = true
					@test participate(fv)[i] == true
				end
			end

			for i in 4:8
				fv[i] = false
				@test fv[i] == false
				@test mk.child.mask[i - 3] == false

				fv[i] = true
				@test fv[i] == true
				@test mk.child.mask[i - 3] == true

				if test_participation
					participate(mk.child.mask)[i - 3] = false
					@test participate(fv)[i] == false
					participate(mk.child.mask)[i - 3] = true
					@test participate(fv)[i] == true
				end
			end

			for i in 9:10 in 
				fv[i] = false
				@test fv[i] == false
				@test mk.child.child[:an].mask[i - 8] == false

				fv[i] = true
				@test fv[i] == true
				@test mk.child.child[:an].mask[i - 8] == true

				if test_participation
					participate(mk.child.child[:an].mask)[i - 8] = false
					@test participate(fv)[i] == false
					participate(mk.child.child[:an].mask)[i - 8] = true
					@test participate(fv)[i] == true
				end
			end

			for i in 11:15 in 
				fv[i] = false
				@test fv[i] == false
				@test mk.child.child[:on].mask[i - 10] == false

				fv[i] = true
				@test fv[i] == true
				@test mk.child.child[:on].mask[i - 10] == true
				if test_participation
					participate(mk.child.child[:on].mask)[i - 10] = false
					@test participate(fv)[i] == false
					participate(mk.child.child[:on].mask)[i - 10] = true
					@test participate(fv)[i] == true
				end
			end
			
			for i in 16:20 in 
				fv[i] = false
				@test fv[i] == false
				@test mk.child.child[:cn].mask[i - 15] == false

				fv[i] = true
				@test fv[i] == true
				@test mk.child.child[:cn].mask[i - 15] == true

				if test_participation
					participate(mk.child.child[:cn].mask)[i - 15] = false
					@test participate(fv)[i] == false
					participate(mk.child.child[:cn].mask)[i - 15] = true
					@test participate(fv)[i] == true
				end
			end
			
			for i in 21:25 in 
				fv[i] = false
				@test fv[i] == false
				@test mk.child.child[:sn].mask[i - 20] == false

				fv[i] = true
				@test fv[i] == true
				@test mk.child.child[:sn].mask[i - 20] == true

				if test_participation
					participate(mk.child.child[:sn].mask)[i - 20] = false
					@test participate(fv)[i] == false
					participate(mk.child.child[:sn].mask)[i - 20] = true
					@test participate(fv)[i] == true
				end
			end
		end

		@testset "broadcasting and setindex in FlatView" begin
			mk = create_mask_structure(ds, mfun)
			fv = FlatView(mk)

			fv .= false
			@test all(fv[i] == false for i in 1:length(fv))

			x = rand([true, false], length(fv))
			fv .= x
			@test all(fv[i] == x[i] for i in 1:length(fv))

			fill!(fv, true)
			@test all(fv[i] == true for i in 1:length(fv))

			x = rand([true, false], length(fv))
			foreach(i -> fv[i] = x[i], 1:length(fv))
			@test all(fv[i] == x[i] for i in 1:length(fv))

			fill!(fv, true)
			@test all(fv[i] == true for i in 1:length(fv))

			# testing access by a vector
			ii = [1,4,9,11,16,21]
			δi = setdiff(1:length(fv), ii) 
			@test_throws ErrorException fv[ii] .= false
			fv[ii] = false
			@test all(fv[i] == false for i in ii)
			@test all(fv[i] == true for i in δi)

			fv .= true
			@test all(fv[i] == true for i in 1:length(fv))
		end
	
		@testset "Testing ignoring of an empty part" begin 
			ds = ProductNode((
				a = BagNode(ArrayNode(NGramMatrix(String[], 3, 256, 2057)), AlignedBags([0:-1, 0:-1])),
				b = ArrayNode(NGramMatrix(["a","b"], 3, 256, 2057)),
				))
			mk = create_mask_structure(ds, mfun)
			fv = FlatView(mk)
			@test length(fv) == 2
			@test ds[mk] == ds
			fv[1] = false
			@test ds[mk][:a] == ds[:a]
			@test isequal(ds[mk][:b].data.S, [missing, "b"])
			fv .= [true, false]
			@test ds[mk][:a] == ds[:a]
			@test isequal(ds[mk][:b].data.S, ["a", missing])
		end
	end	
end

@testset "Parental structure" begin
	ds = specimen_sample()
	mk = create_mask_structure(ds, d -> SimpleMask(fill(true, d)))
	all_mk = IdDict(ExplainMill.collect_masks_with_levels(mk))
	@test all_mk[mk.mask] == 1
	@test all_mk[mk.child.mask] == 2
	for s in keys(mk.child.child)
		@test all_mk[mk.child.child[s].mask] == 3
	end

	# test that in presence of multiple observation masks,
	# there will be only on
	@test length(ExplainMill.collect_masks_with_levels(mk)) == 6
	@test length(ExplainMill.collectmasks(mk)) == 6
	@set! mk.child.child.childs.an = ObservationMask(SimpleMask(fill(true, 5)))
	@set! mk.child.child.childs.cn = mk.child.child.childs.an
	@set! mk.child.child.childs.on = mk.child.child.childs.an
	@set! mk.child.child.childs.sn = mk.child.child.childs.an
	@test length(ExplainMill.collect_masks_with_levels(mk)) == 3
	@test length(ExplainMill.collectmasks(mk)) == 3
end
