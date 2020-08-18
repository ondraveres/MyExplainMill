using ExplainMill, Mill, JsonGrinder
using ExplainMill: Mask, yarason, participate, prunemask, addor, OR, EmptyMask
using Setfield
using Test
using SparseArrays

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

		c = participate(m) .& prunemask(m)
		@test addor(m, xs[c], c) == [OR(["a", "e"]), OR(["b", "d"]), "c", OR(["b", "d"]), OR(["a", "e"]),]
		m.mask[1] = false
		c = participate(m) .& prunemask(m)
		@test addor(m, xs[c], c) == [OR(["b", "d"]), "c", OR(["b", "d"])]
		m.mask[1:2] = [true,false]
		c = participate(m) .& prunemask(m)
		@test addor(m, xs[c], c) == [OR(["a", "e"]), "c", OR(["a", "e"]),]
		m.mask[2:3] = [true, false]
		c = participate(m) .& prunemask(m)
		@test addor(m, xs[c], c) == [OR(["a", "e"]), OR(["b", "d"]), OR(["b", "d"]), OR(["a", "e"])]

		m = ExplainMill.Mask(5, d -> zeros(d))
		m.mask[1] = false
		c = participate(m) .& prunemask(m)
		@test addor(m, xs[c], c) == xs[participate(m) .& prunemask(m)]
	end

	@testset "Matrix" begin
		an = ArrayNode([1 0 3 0 5; 1 2 0 4 0])
		am = Mask(an, d -> rand(d))

		@test matcharrays(yarason(an, am, nothing),  [[1, 1], [0, 2], [3, 0], [0, 4], [5, 0]])
		@test matcharrays(yarason(an, EmptyMask(), nothing),  [[1, 1], [0, 2], [3, 0], [0, 4], [5, 0]])
		am.mask.mask[1] = false
		@test matcharrays(yarason(an, am, nothing),   [[1], [2], [0], [4], [0]])
		@test matcharrays(yarason(an, am, nothing, [true, false, true, false, true]),   [[1], [0], [0]])
		@test matcharrays(yarason(an, EmptyMask(), nothing, [true, false, true, false, true]),   [[1, 1], [3, 0],  [5, 0]])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, nothing),   [[], [], [], [], []])
	end

	@testset "Sparse" begin
		an = ArrayNode(sparse([1 0 3 0 5; 1 2 0 4 0]))
		am = Mask(an, d -> rand(d))

		@test_broken matcharrays(yarason(an, am, nothing),  [[1, 1], [0, 2], [3, 0], [0, 4], [5, 0]])
		am.mask.mask[1] = false
		@test_broken matcharrays(yarason(an, am, nothing),   [[1], [2], [0], [4], [0]])
		@test_broken matcharrays(yarason(an, am, nothing, [true, false, true, false, true]),   [[1], [0], [0]])
	end

	@testset "NGramMatrix" begin
		an = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
		am = Mask(an, d -> rand(d))
		@test matcharrays(yarason(an, am, nothing), ["a","b","c","d","e"])
		@test matcharrays(yarason(an, EmptyMask(), nothing), ["a","b","c","d","e"])
		@test matcharrays(yarason(an, EmptyMask(), nothing, [true, false, true,false,false]), ["a","c"])
		am.mask.mask[[2,4]] .= false
		@test matcharrays(yarason(an, am, nothing), ["a", missing, "c", missing, "e"])
		@test matcharrays(yarason(an, am, nothing, [true,false,true,false,true]), ["a","c","e"])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e) ,  [missing, missing, missing, missing, missing])
		@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , [missing, missing])

		am = @set am.mask = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))
		am.mask.mask[2] = false
		@test matcharrays(yarason(an, am, nothing), [OR(["a", "e"]), missing, "c", missing, OR(["a", "e"])])
		@test matcharrays(yarason(an, am, nothing, [true, true, true,false,false]), ["a", missing, "c"])
		@test matcharrays(yarason(an, am, nothing, [false, false, false,false,false]), [])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e) ,  [missing, missing, missing, missing, missing])
		@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , [missing, missing])
	end

	@testset "Categorical" begin
		e = ExtractCategorical(["a","b","c","d"])
		an = e(["a","b","c","d","e"])
		am = Mask(an, d -> rand(d))
		@test matcharrays(yarason(an, am, e) , ["a","b","c","d","__UNKNOWN__"])
		@test matcharrays(yarason(an, EmptyMask(), e), ["a","b","c","d","__UNKNOWN__"])
		@test matcharrays(yarason(an, EmptyMask(), e, [true, false, true,false,false]), ["a","c"])

		am.mask.mask[[2,4]] .= false
		@test matcharrays(yarason(an, am, e) , ["a",missing, "c", missing, "__UNKNOWN__"])
		@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , ["a", missing])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e) ,  [missing, missing, missing, missing, missing])
		@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , [missing, missing])

		am = @set am.mask = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))
		am.mask.mask[2] = false
		@test matcharrays(yarason(an, am, e) , [OR(["a","__UNKNOWN__"]), missing, "c", missing, OR(["a","__UNKNOWN__"])])
		@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , ["a", missing])
		@test matcharrays(yarason(an, am, e, [false, false, false, false, false]) , [])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e) ,  [missing, missing, missing, missing, missing])
	end

	@testset "Lazy" begin
		e = nothing
		an = LazyNode(:Test,["a","b","c","d","e"])
		am = Mask(an, d -> rand(d))


		@test matcharrays(yarason(an, am, e) , ["a","b","c","d","e"])
		@test matcharrays(yarason(an, am, e, [true, false, true, false, false]) , ["a","c"])
		@test matcharrays(yarason(an, am, e, [false, false, false, false, false]) , [])
	end
end
