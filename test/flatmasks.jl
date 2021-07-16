using ExplainMill
using Test
using Mill
using Flux
using SparseArrays
using Random
using ExplainMill: Mask, FlatView		
using ExplainMill: ParticipationTracker, create_mask_structure, participate
include("specimen.jl")
@testset "mapping between flat structure and nodes" begin
	ds = specimen_sample()
	for mfun in [
		d -> SimpleMask(fill(true, d)),
		d -> HeuristicMask(ones(Float32, d)),
		d -> ParticipationTracker(SimpleMask(fill(true, d))),
		d -> ParticipationTracker(HeuristicMask(ones(Float32, d))),
		]

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
		end
	
		@testset "Parental structure" begin
			ds = specimen_sample()
			
		end
	end	
end

