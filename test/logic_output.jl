using ExplainMill, Mill, JsonGrinder
using ExplainMill: Mask, yarason, participate, prunemask, addor, OR
using Setfield
using Test

function matcharrays(a, b)
	(length(a) != length(b)) && return(false)
	for i in 1:length(a)
		ismissing(a[i]) && ismissing(b[i]) && continue
		ismissing(a[i]) && return(false)
		ismissing(b[i]) && return(false)
		(a[i] != b[i]) && return(false)
	end
	return(true)
end

@testset "logical output" begin 
	@testset "OR relationship" begin 
		xs = ["a","b","c","d","e"]
		m = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))

		@test addor(m, xs[participate(m) .& prunemask(m)]) == [OR(["a", "e"]), OR(["b", "d"]), ["c"], OR(["b", "d"]), OR(["a", "e"]),]
		m.mask[1] = false
		@test addor(m, xs[participate(m) .& prunemask(m)]) == [OR(["b", "d"]), ["c"], OR(["b", "d"])]
		m.mask[1:2] = [true,false]
		@test addor(m, xs[participate(m) .& prunemask(m)]) == [OR(["a", "e"]), ["c"], OR(["a", "e"]),]
		m.mask[2:3] = [true, false]
		@test addor(m, xs[participate(m) .& prunemask(m)]) == [OR(["a", "e"]), OR(["b", "d"]), OR(["b", "d"]), OR(["a", "e"])]

		m = ExplainMill.Mask(5, d -> zeros(d))
		m.mask[1] = false
		@test addor(m, xs[participate(m) .& prunemask(m)]) == xs[participate(m) .& prunemask(m)]
	end

# @testset "Testing correctness of detecting samples that should not be considered in the calculation of daf values" begin
# 	an = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
# 	am = Mask(an, d -> rand(d))
# 	m, ds = am, an
# 	all(yarason(an, am, nothing) .== ["a","b","c","d","e"])
# 	all(yarason(an, am, nothing) .== ["a","b","c","d","e"])

# 	cn = ArrayNode(sparse([1 0 3 0 5; 1 2 0 4 0]))
	
# 	ds = BagNode(BagNode(sn, AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))
# 	ds = BagNode(ProductNode((a = cn, b = sn)), AlignedBags([1:2,3:3,4:5]))
# end

@testset "NGramMatrix" begin
	an = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
	am = Mask(an, d -> rand(d))
	@test matcharrays(yarason(an, am, nothing), ["a","b","c","d","e"])
	am.mask.mask[[2,4]] .= false
	@test all(yarason(an, am, nothing) .== ["a","c","e"])

	am = @set am.mask = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))
	am.mask.mask[2] = false
	@test all(yarason(an, am, nothing) .== [OR(["a", "e"]), ["c"], OR(["a", "e"])])
end

# below is the version, where "unknown keys are not exported"
# @testset "Categorical" begin
# 	e = ExtractCategorical(["a","b","c","d"])
# 	an = e(["a","b","c","d","e"])
# 	am = Mask(an, d -> rand(d))
# 	@test all(yarason(an, am, e) .== ["a","b","c","d"])
# 	am.mask.mask[[2,4]] .= false
# 	@test all(yarason(an, am, e) .== ["a","c"])

# 	am = @set am.mask = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))
# 	am.mask.mask[2] = false
# 	@test all(yarason(an, am, e) .== [["a"], ["c"]])
# end

@testset "Categorical" begin
	e = ExtractCategorical(["a","b","c","d"])
	an = e(["a","b","c","d","e"])
	am = Mask(an, d -> rand(d))
	@test matcharrays(yarason(an, am, e) , ["a","b","c","d","__UNKNOWN__"])
	am.mask.mask[[2,4]] .= false
	@test matcharrays(yarason(an, am, e) , ["a",missing, "c", missing, "__UNKNOWN__"])
	@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , ["a", missing])

	am = @set am.mask = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))
	am.mask.mask[2] = false
	@test matcharrays(yarason(an, am, e) , [OR(["a","__UNKNOWN__"]), missing, ["c"], missing, OR(["a","__UNKNOWN__"])])
	@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , [["a"], missing])
end

@testset "Lazy" begin
	e = nothing
	an = LazyNode(:Test,["a","b","c","d","e"])
	am = Mask(an, d -> rand(d))


	@test matcharrays(yarason(an, am, e) , ["a","b","c","d","__UNKNOWN__"])
	am.mask.mask[[2,4]] .= false
	@test matcharrays(yarason(an, am, e) , ["a","c","__UNKNOWN__"])

	am = @set am.mask = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))
	am.mask.mask[2] = false
	@test matcharrays(yarason(an, am, e) , [OR(["a","__UNKNOWN__"]), ["c"], OR(["a","__UNKNOWN__"])])
end

