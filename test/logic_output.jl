using ExplainMill, Mill, JsonGrinder
using ExplainMill: Mask, yarason, participate, prunemask, addor, OR, EmptyMask, logicaland
using Setfield
using Test
using SparseArrays

const SN = Union{String,T} where T<:Number
matcharrays(a::Missing, b::Missing) = true
matcharrays(a::Missing, b::SN) = false
matcharrays(a, b) = a == b
function matcharrays(a::Vector, b::Vector)
	(length(a) != length(b)) && return(false)
	all(matcharrays(a[i],b[i]) for i in 1:length(a))
end

function matcharrays(a::Dict, b::Dict)
	ks = union(keys(a), keys(b))
	!isempty(setdiff(ks,keys(a))) && (@info "different keys";return(false))
	!isempty(setdiff(ks,keys(b))) && (@info "different keys";return(false))
	for i in ks
		if !matcharrays(a[i],b[i])
			@info "different in key $i"
			return(false)
		end
	end
	true
end

@testset "logical output" begin 
	@testset "OR relationship" begin 
		xs = ["a","b","c","d","e"]
		m = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))

		c = participate(m) .& prunemask(m)
		@test matcharrays(addor(m, xs[c], c), [OR(["a", "e"]), OR(["b", "d"]), "c", OR(["b", "d"]), OR(["a", "e"])])
		m.mask[1] = false
		c = participate(m) .& prunemask(m)
		@test matcharrays(addor(m, xs[c], c), [OR(["b", "d"]), "c", OR(["b", "d"])])
		m.mask[1:2] = [true,false]
		c = participate(m) .& prunemask(m)
		@test matcharrays(addor(m, xs[c], c), [OR(["a", "e"]), "c", OR(["a", "e"])])
		m.mask[2:3] = [true, false]
		c = participate(m) .& prunemask(m)
		@test matcharrays(addor(m, xs[c], c), [OR(["a", "e"]), OR(["b", "d"]), OR(["b", "d"]), OR(["a", "e"])])

		m = ExplainMill.Mask(5, d -> zeros(d))
		m.mask[1] = false
		c = participate(m) .& prunemask(m)
		@test addor(m, xs[c], c) == xs[participate(m) .& prunemask(m)]
	end

	@testset "Matrix" begin
		e = ExtractScalar(Float32, 0, 1)
		an = ArrayNode([1 0 3 0 5; 1 2 0 4 0])
		am = Mask(an, d -> rand(d))

		@test matcharrays(yarason(an, am, e),  [[1, 1], [0, 2], [3, 0], [0, 4], [5, 0]])
		@test matcharrays(yarason(an, EmptyMask(), e),  [[1, 1], [0, 2], [3, 0], [0, 4], [5, 0]])
		am.mask.mask[1] = false
		@test matcharrays(yarason(an, am, e),   [[missing, 1], [missing, 2], [missing, 0], [missing, 4], [missing, 0]])
		@test matcharrays(yarason(an, am, e, [true, false, true, false, true]),   [[missing,1], [missing, 0], [missing, 0]])
		@test matcharrays(yarason(an, EmptyMask(), nothing, [true, false, true, false, true]),   [[1, 1], [3, 0],  [5, 0]])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e),   fill([missing, missing], 5))

		e = ExtractScalar(Float32, 1, 2)
		am.mask.mask .= true
		@test matcharrays(yarason(an, am, e),   [[1.5, 1.5], [1.0, 2.0], [2.5, 1.0], [1.0, 3.0], [3.5, 1.0]])
		am.mask.mask[1] = false
		@test matcharrays(yarason(an, am, e),   [[missing, 1.5], [missing, 2.0], [missing, 1.0], [missing, 3.0], [missing, 1.0]])
	end

	@testset "NGramMatrix" begin
		e = ExtractString(String)
		an = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
		am = Mask(an, d -> rand(d))
		@test matcharrays(yarason(an, am, e), ["a","b","c","d","e"])
		@test matcharrays(yarason(an, EmptyMask(), e), ["a","b","c","d","e"])
		@test matcharrays(yarason(an, EmptyMask(), e, [true, false, true,false,false]), ["a","c"])
		am.mask.mask[[2,4]] .= false
		@test matcharrays(yarason(an, am, e), ["a", missing, "c", missing, "e"])
		@test matcharrays(yarason(an, am, e, [true,false,true,false,true]), ["a","c","e"])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e) ,  [missing, missing, missing, missing, missing])
		@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , [missing, missing])

		am = @set am.mask = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))
		am.mask.mask[2] = false
		@test matcharrays(yarason(an, am, e), [OR(["a", "e"]), missing, "c", missing, OR(["a", "e"])])
		@test matcharrays(yarason(an, am, e, [true, true, true,false,false]), ["a", missing, "c"])
		@test matcharrays(yarason(an, am, e, [false, false, false,false,false]), [])

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

	@testset "Bags" begin
		e = ExtractCategorical(["a","b","c","d"])
		e = ExtractArray(e)
		an = reduce(catobs, e.([["a","b"],["c"],["d","e"]]))
		am = Mask(an, d -> rand(d))

		@test matcharrays(yarason(an, am, e), [["a", "b"], ["c"], ["d", "__UNKNOWN__"]])
		@test matcharrays(yarason(an, am, e, [true, false, true]) , [["a", "b"], ["d", "__UNKNOWN__"]])
		@test matcharrays(yarason(an, am, e, [false, true, false]) , [["c"]])

		@test matcharrays(yarason(an, EmptyMask(), e), [["a", "b"], ["c"], ["d", "__UNKNOWN__"]])
		@test matcharrays(yarason(an, EmptyMask(), e, [true, false, true]) , [["a", "b"], ["d", "__UNKNOWN__"]])
		@test matcharrays(yarason(an, EmptyMask(), e, [false, true, false]) , [["c"]])

		am.mask.mask .= [true,false,true,false,true]
		@test matcharrays(yarason(an, am, e) , [["a"], ["c"], ["__UNKNOWN__"]])
		@test matcharrays(yarason(an, am, e, [true, false, true]) , [["a"], ["__UNKNOWN__"]])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e) , fill([], 3))
		@test matcharrays(yarason(an, am, e, [true, false, true]) , fill([], 2))

		am.mask.mask .= true
		am.child.mask.mask .= [true,false,true,false,true]
		@test matcharrays(yarason(an, am, e) , [["a", missing], ["c"], [missing, "__UNKNOWN__"]])
		@test matcharrays(yarason(an, am, e, [true, false, true]) , [["a", missing], [missing, "__UNKNOWN__"]])
		am.mask.mask .= [true,false,true,false,true]
		@test matcharrays(yarason(an, am, e) , [["a"], ["c"], ["__UNKNOWN__"]])
		@test matcharrays(yarason(an, am, e, [true, false, true]) , [["a"], ["__UNKNOWN__"]])
		
		am.mask.mask .= true
		am.child.mask.mask .= false
		@test matcharrays(yarason(an, am, e) , [[missing], [missing], [missing]])
		@test matcharrays(yarason(an, am, e, [true, false, true]) , [[missing], [missing]])

		an = BagNode(ArrayNode(zeros(2,0)), fill(0:-1, 3))
		am = Mask(an, d -> rand(d))
		@test matcharrays(yarason(an, am, e), fill(missing, 3))
	end

	@testset "Product" begin
		e = ExtractDict(nothing, Dict(:a => ExtractCategorical(["ca","cb","cc","cd"]), 
			:b => ExtractString(String)))
		s = map(x -> Dict(:a => x[1], :b => x[2]), zip(["ca","cb","cc","cd","ce"], ["sa","sb","sc","sd","se"]))
		an = reduce(catobs,e.(s))
		am = Mask(an, d -> rand(d))

		expected = [Dict(:a => "ca",:b => "sa"), Dict(:a => "cb",:b => "sb"), Dict(:a => "cc",:b => "sc"), Dict(:a => "cd",:b => "sd"), Dict(:a => "__UNKNOWN__",:b => "se")]
		@test matcharrays(yarason(an, am, e) , expected)
		@test matcharrays(yarason(an, EmptyMask(), e) , expected)
		@test matcharrays(yarason(an, am, e, [true, false, true,false,false]),  expected[[1,3]])
		@test matcharrays(yarason(an, EmptyMask(), e, [true, false, true,false,false]),  expected[[1,3]])

		am[:a].mask.mask[[2,4]] .= false
		@test matcharrays(yarason(an, am, e) , [Dict(:a => "ca",:b => "sa"), Dict(:a => missing,:b => "sb"), Dict(:a => "cc",:b => "sc"), Dict(:a => missing,:b => "sd"), Dict(:a => "__UNKNOWN__",:b => "se")])
		am[:b].mask.mask[[2,4]] .= false
		@test matcharrays(yarason(an, am, e) , [Dict(:a => "ca",:b => "sa"), Dict(:a => missing,:b => missing), Dict(:a => "cc",:b => "sc"), Dict(:a => missing,:b => missing), Dict(:a => "__UNKNOWN__",:b => "se")])
		am[:a].mask.mask .= true
		@test matcharrays(yarason(an, am, e) , [Dict(:a => "ca",:b => "sa"), Dict(:a => "cb",:b => missing), Dict(:a => "cc",:b => "sc"), Dict(:a => "cd",:b => missing), Dict(:a => "__UNKNOWN__",:b => "se")])

		am[:a].mask.mask .= true
		am[:b].mask.mask .= true
		am[:a].mask.mask[[2,4]] .= false
		@test matcharrays(yarason(an, am, e, [true, false, false, true, false]) ,[Dict(:a => "ca",:b => "sa"), Dict(:a => missing,:b => "sd")])
		am[:b].mask.mask[[2,4]] .= false
		@test matcharrays(yarason(an, am, e, [true, false, false, true, false]) , [Dict(:a => "ca",:b => "sa"), Dict(:a => missing,:b => missing)])
		@test matcharrays(yarason(an, am, e, [true, false, true, false, false]) ,  [Dict(:a => "ca",:b => "sa"), Dict(:a => "cc",:b => "sc")])
	end
	
	@testset "logicaland" begin
		@test logicaland(1, missing) == 1
		@test logicaland(missing, 1) == 1
		@test ismissing(logicaland(missing, missing))
		@test logicaland([1,2], [1]) == [1]
		@test logicaland([1], [1,2]) == [1]
		@test logicaland(1, OR([1,2])) == 1
		@test logicaland(OR([1,3]), OR([1,2])) == 1
		@test logicaland(OR([1,2]), OR([1,2])) == OR([1,2])
		@test logicaland("__UNKNOWN__","a") == "a"
		@test logicaland("a", "__UNKNOWN__") == "a"
		@test logicaland("__UNKNOWN__", "__UNKNOWN__") == "__UNKNOWN__"
	end

	@testset "MultipleRepresentation" begin
		e = MultipleRepresentation(
			(ExtractCategorical(["a", "b","c","d"]),
			JsonGrinder.ExtractString(String))
			)

		s = ["a", "b", "c", "d", "e"]
		an = reduce(catobs,e.(s))
		am = Mask(an, d -> rand(d))

		@test matcharrays(yarason(an, am, e) , s)
	end

	@testset "ExtractKeyAsField" begin
		e = JsonGrinder.ExtractKeyAsField(
			JsonGrinder.ExtractString(String),
			JsonGrinder.ExtractArray(
				JsonGrinder.ExtractString(String)),
		)
		s = [Dict(
			"ka" => ["a1", "a2", "a3"],
			),
			Dict(
			"kc" => ["c1"],
			"ka" => ["a1", "a2"],
			),]
		
		an = reduce(catobs,e.(s))
		am = Mask(an, d -> rand(d))

		@test matcharrays(yarason(an, am, e), [[Dict("ka" => ["a1", "a2", "a3"])], [Dict("ka" => ["a1", "a2"]), Dict("kc" => ["c1"])]])
		@test matcharrays(yarason(an, am, e, [true, false]) , [[Dict("ka" => ["a1", "a2", "a3"])]])
		am.mask.mask .= [false, true, true]
		@test matcharrays(yarason(an, am, e), [[], [Dict("ka" => ["a1", "a2"]), Dict("kc" => ["c1"])]])
		am.mask.mask .= [true, false, true]
		@test matcharrays(yarason(an, am, e),  [[Dict("ka" => ["a1", "a2", "a3"])], [Dict("kc" => ["c1"])]])
		am.mask.mask .= true
		am.mask.mask .= [true, false, true]
		am.child[:key].mask.mask .=[false, true, true]
		@test matcharrays(yarason(an, am, e),  [[Dict(missing => ["a1", "a2", "a3"])], [Dict("kc" => ["c1"])]])
	end 
	
	
end
