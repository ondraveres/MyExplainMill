using ExplainMill
using Mill
using Test
using Flux
using SparseArrays
using ExplainMill: collectmasks
using Mill: partialeval


@testset "Partial Evaluation" begin 
    ds = specimen_sample()
    model = reflectinmodel(ds, d -> Dense(d, 4), SegmentedMean)

	for mfun in [
		d -> SimpleMask(fill(true, d)),
		d -> ParticipationTracker(SimpleMask(fill(true, d))),
		]

	    mk = create_mask_structure(ds, mfun)

	    @testset "topmost bagmask" begin
	    	for mr in [mk, mk.mask]
			    mₚ, dsₚ, mkₚ, keep = partialeval(model, ds, mk, [mr])
			    @test mₚ.im.m == identity
			    @test dsₚ.data isa ArrayNode
			    @test mkₚ.child isa EmptyMask
			    @test keep
		    	@test model(ds[mk]).data ≈ mₚ(dsₚ[mkₚ]).data
			    for i in 1:length(mkₚ.mask)
			    	mkₚ.mask[i] = false
			    	@test model(ds[mk]).data ≈ mₚ(dsₚ[mkₚ]).data
			    	@test model(ds, mk).data ≈ mₚ(dsₚ, mkₚ).data
			    	mkₚ.mask[i] = true
			    end
			end
		end

	    @testset "topmost bagmask" begin
	    	for mr in [mk.child, mk.child.mask]
			    mₚ, dsₚ, mkₚ, keep = partialeval(model, ds, mk, [mr])
			    @test mₚ.im.im.m == identity
			    @test dsₚ.data.data isa ArrayNode
			    @test mkₚ.child.child isa EmptyMask
			    @test keep
		    	@test model(ds[mk]).data ≈ mₚ(dsₚ[mkₚ]).data
		    	fv = FlatView(mkₚ)
			    for i in 1:length(fv)
			    	fv[i] = false
			    	@test model(ds[mk]).data ≈ mₚ(dsₚ[mkₚ]).data
			    	@test model(ds, mk).data ≈ mₚ(dsₚ, mkₚ).data
			    	fv[i] = true
			    end
			end
		end

		@testset "bottom productmasks" begin
			for k in keys(mk.child.child)
		    	for mr in [mk.child.child[k], mk.child.child[k].mask]
				    mₚ, dsₚ, mkₚ, keep = partialeval(model, ds, mk, [mr])
				    for l in setdiff(keys(mk.child.child), [k])
					    @test mₚ.im.im[l].m == identity
					    @test dsₚ.data.data[l] isa ArrayNode
					    @test mkₚ.child.child[l] isa EmptyMask
					end
				    @test keep
			    	@test model(ds[mk]).data ≈ mₚ(dsₚ[mkₚ]).data
			    	fv = FlatView(mkₚ)
				    for i in 1:length(fv)
				    	fv[i] = false
				    	@test model(ds[mk]).data ≈ mₚ(dsₚ[mkₚ]).data
				    	@test model(ds, mk).data ≈ mₚ(dsₚ, mkₚ).data
				    	fv[i] = true
				    end
				end
			end
		end
	end
end