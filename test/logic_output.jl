####
# Unit test for extraction of logical formulas and their matching
####
using ExplainMill, Mill, JsonGrinder
using ExplainMill: Mask, yarason, participate, prunemask, addor, OR, EmptyMask, logicaland
using ExplainMill: Absent, absent, isabsent
using Setfield
using Test
using SparseArrays

const SN = Union{String,T} where T<:Number

matcharrays(a::Absent, b::Absent) = true
matcharrays(a::Absent, b::SN) = false
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

	@testset "Matrix - Rules" begin
		e = ExtractScalar(Float32, 0, 1)
		an = ArrayNode([1 0 3 0 5; 1 2 0 4 0])
		am = Mask(an, d -> rand(d))

		y = yarason(an, am, e)
		@test matcharrays(y,  [[1, 1], [0, 2], [3, 0], [0, 4], [5, 0]])
		@test Base.match(an, [[1, 1], [0, 2]], e)
		@test !Base.match(an[1], [[1, 1], [0, 2]], e)
		@test Base.match(an, [[1, 1]], e)
		@test Base.match(an, [[absent, 1]], e)
		@test Base.match(an, [[absent, absent]], e)
		@test Base.match(an, absent, e)
		@test Base.match(an, [absent], e)

		@test matcharrays(yarason(an, EmptyMask(), e),  [[1, 1], [0, 2], [3, 0], [0, 4], [5, 0]])
		am.mask.mask[1] = false
		@test matcharrays(yarason(an, am, e),   [[absent, 1], [absent, 2], [absent, 0], [absent, 4], [absent, 0]])
		@test matcharrays(yarason(an, am, e, [true, false, true, false, true]),   [[absent,1], [absent, 0], [absent, 0]])
		@test matcharrays(yarason(an, EmptyMask(), nothing, [true, false, true, false, true]),   [[1, 1], [3, 0],  [5, 0]])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e),   fill([absent, absent], 5))

		e = ExtractScalar(Float32, 1, 2)
		am.mask.mask .= true
		@test matcharrays(yarason(an, am, e),   [[1.5, 1.5], [1.0, 2.0], [2.5, 1.0], [1.0, 3.0], [3.5, 1.0]])
		am.mask.mask[1] = false
		@test matcharrays(yarason(an, am, e),   [[absent, 1.5], [absent, 2.0], [absent, 1.0], [absent, 3.0], [absent, 1.0]])
	end

	@testset "NGramMatrix" begin
		e = ExtractString(String)
		s = ["a","b","c","d","e"]
		ss = ["z","b","c","d","e"]
		an = reduce(catobs, e.(s))
		az = reduce(catobs, e.(ss))
		am = Mask(an, d -> rand(d))

		y = yarason(an, am, e)
		@test matcharrays(y, ["a","b","c","d","e"])
		@test Base.match(an, y, e)	
		@test !Base.match(az, y, e)	

		y = yarason(an, EmptyMask(), e)
		@test matcharrays(y, ["a","b","c","d","e"])
		@test Base.match(an, y, e)	
		@test !Base.match(az, y, e)	

		y = yarason(an, EmptyMask(), e, [true, false, true,false,false])
		@test matcharrays(y, ["a","c"])
		@test Base.match(an, y, e)	
		@test !Base.match(az, y, e)	

		am.mask.mask[[2,4]] .= false
		y = yarason(an, am, e)
		@test matcharrays(y, ["a", absent, "c", absent, "e"])
		@test Base.match(an, y, e)
		y = yarason(an, am, e, [true,false,true,false,true])
		@test matcharrays(y, ["a","c","e"])
		@test Base.match(an, y, e)	
		@test !Base.match(az, y, e)	

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e) ,  [absent, absent, absent, absent, absent])
		@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , [absent, absent])

		am = @set am.mask = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))
		am.mask.mask[2] = false
		y = yarason(an, am, e)
		@test matcharrays(y, [OR(["a", "e"]), absent, "c", absent, OR(["a", "e"])])
		@test !match(e("a"), y, e)
		@test match(e("a"), y[1], e)
		@test match(e("a"), y[1:2], e)
		@test !match(e("a"), y[1:3], e)

		@test matcharrays(yarason(an, am, e, [true, true, true,false,false]), ["a", absent, "c"])
		@test matcharrays(yarason(an, am, e, [false, false, false,false,false]), [])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e) ,  [absent, absent, absent, absent, absent])
		@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , [absent, absent])
	end

	@testset "Categorical" begin
		e = ExtractCategorical(["a","b","c","d"])
		an = e(["a","b","c","d","e"])
		am = Mask(an, d -> rand(d))

		y = yarason(an, am, e)
		@test matcharrays(y , ["a","b","c","d","__UNKNOWN__"])
		@test Base.match(an, y, e)	
		@test matcharrays(yarason(an, EmptyMask(), e), ["a","b","c","d","__UNKNOWN__"])

		y = yarason(an, EmptyMask(), e, [true, false, true,false,false])
		@test matcharrays(y, ["a","c"])
		@test Base.match(an, y, e)	
		@test !Base.match(an[2:end], y, e)	

		am.mask.mask[[2,4]] .= false
		y = yarason(an, am, e)
		@test matcharrays(y , ["a",absent, "c", absent, "__UNKNOWN__"])
		@test Base.match(an, y, e)	
		@test !Base.match(an[2:end], y, e)	
		@test !Base.match(an[1:end-1], y, e)	
		@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , ["a", absent])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e) ,  [absent, absent, absent, absent, absent])
		@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , [absent, absent])

		am = @set am.mask = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))
		am.mask.mask[2] = false
		y = yarason(an, am, e)
		@test matcharrays(y , [OR(["a","__UNKNOWN__"]), absent, "c", absent, OR(["a","__UNKNOWN__"])])
		@test !match(e("a"), y, e)
		@test match(e("a"), y[1], e)
		@test match(e("a"), y[1:2], e)
		@test !match(e("a"), y[1:3], e)
		@test !match(e("c"), y[1:3], e)
		@test match(e("f"), y[1], e)

		@test matcharrays(yarason(an, am, e, [true, true, false, false, false]) , ["a", absent])
		@test matcharrays(yarason(an, am, e, [false, false, false, false, false]) , [])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e) ,  [absent, absent, absent, absent, absent])
	end

	@testset "Lazy" begin
		e = nothing
		an = LazyNode(:Test,["a","b","c","d","e"])
		am = Mask(an, d -> rand(d))

		y = yarason(an, am, e)
		@test matcharrays(y , ["a","b","c","d","e"])
		@test Base.match(an, y, e)	
		@test !Base.match(an[2:end], y, e)	
		@test Base.match(an[2:end], y[2:end], e)	
		@test Base.match(an[2:end], [], e)	

		@test matcharrays(yarason(an, am, e, [true, false, true, false, false]) , ["a","c"])
		@test matcharrays(yarason(an, am, e, [false, false, false, false, false]) , [])
	end

	@testset "Bags" begin
		e = ExtractCategorical(["a","b","c","d"])
		e = ExtractArray(e)
		an = reduce(catobs, e.([["a","b"],["c"],["d","e"]]))
		am = Mask(an, d -> rand(d))

		y = yarason(an, am, e)
		@test matcharrays(y, [["a", "b"], ["c"], ["d", "__UNKNOWN__"]])
		@test Base.match(an, y, e)	
		@test Base.match(an, y[1:2], e)
		@test !Base.match(an[1:2], y, e)
		@test matcharrays(yarason(an, am, e, [true, false, true]) , [["a", "b"], ["d", "__UNKNOWN__"]])
		y = yarason(an, am, e, [false, true, false]) 
		@test matcharrays(y , [["c"]])
		@test Base.match(an, y, e)	
		@test !Base.match(an[1], y, e)

		am = @set am.mask = ExplainMill.Mask([1,2,3,2,1], d -> zeros(d))
		y = yarason(an, am, e)
		@test matcharrays(y, [[OR(["a", "__UNKNOWN__"]), OR(["b", "d"])], ["c"], [OR(["b", "d"]), OR(["a", "__UNKNOWN__"])]])
		@test Base.match(an, y, e)	
		@test !Base.match(an[1], y, e)	
		@test Base.match(an[1], y[[1]], e)	
		@test !Base.match(an[2], y[[1]], e)	


		@test matcharrays(yarason(an, EmptyMask(), e), [["a", "b"], ["c"], ["d", "__UNKNOWN__"]])
		@test matcharrays(yarason(an, EmptyMask(), e, [true, false, true]) , [["a", "b"], ["d", "__UNKNOWN__"]])
		@test matcharrays(yarason(an, EmptyMask(), e, [false, true, false]) , [["c"]])

		am = Mask(an, d -> rand(d))
		am.mask.mask .= [true,false,true,false,true]
		@test matcharrays(yarason(an, am, e) , [["a"], ["c"], ["__UNKNOWN__"]])
		@test matcharrays(yarason(an, am, e, [true, false, true]) , [["a"], ["__UNKNOWN__"]])

		am.mask.mask .= false
		@test matcharrays(yarason(an, am, e) , fill([], 3))
		@test matcharrays(yarason(an, am, e, [true, false, true]) , fill([], 2))

		am.mask.mask .= true
		am.child.mask.mask .= [true,false,true,false,true]
		y = yarason(an, am, e)
		@test matcharrays(y , [["a", absent], ["c"], [absent, "__UNKNOWN__"]])
		@test Base.match(an, y, e)	
		@test !Base.match(an[1], y, e)
		@test !Base.match(an[[1,3]], y, e)

		@test matcharrays(yarason(an, am, e, [true, false, true]) , [["a", absent], [absent, "__UNKNOWN__"]])
		am.mask.mask .= [true,false,true,false,true]
		@test matcharrays(yarason(an, am, e) , [["a"], ["c"], ["__UNKNOWN__"]])
		@test matcharrays(yarason(an, am, e, [true, false, true]) , [["a"], ["__UNKNOWN__"]])
		
		am.mask.mask .= true
		am.child.mask.mask .= false
		@test matcharrays(yarason(an, am, e) , [[absent], [absent], [absent]])
		@test matcharrays(yarason(an, am, e, [true, false, true]) , [[absent], [absent]])

		an = BagNode(ArrayNode(zeros(2,0)), fill(0:-1, 3))
		am = Mask(an, d -> rand(d))
		@test matcharrays(yarason(an, am, e), fill(absent, 3))
	end

	@testset "Product" begin
		e = ExtractDict(nothing, Dict(:a => ExtractCategorical(["ca","cb","cc","cd"]), 
			:b => ExtractString(String)))
		s = map(x -> Dict(:a => x[1], :b => x[2]), zip(["ca","cb","cc","cd","ce"], ["sa","sb","sc","sd","se"]))
		an = reduce(catobs,e.(s))
		am = Mask(an, d -> rand(d))

		expected = [Dict(:a => "ca",:b => "sa"), Dict(:a => "cb",:b => "sb"), Dict(:a => "cc",:b => "sc"), Dict(:a => "cd",:b => "sd"), Dict(:a => "__UNKNOWN__",:b => "se")]
		y = yarason(an, am, e)
		@test matcharrays(y, expected)
		@test Base.match(an, y[1], e)	
		@test !Base.match(an[2], y[1], e)	
		@test Base.match(an[1], Dict(:a => "ca"), e)
		@test !Base.match(an[2], Dict(:a => "ca"), e)
		@test_throws String  Base.match(an[1], Dict(:c => "ca"), e)

		@test matcharrays(yarason(an, EmptyMask(), e) , expected)
		@test matcharrays(yarason(an, am, e, [true, false, true,false,false]),  expected[[1,3]])
		@test matcharrays(yarason(an, EmptyMask(), e, [true, false, true,false,false]),  expected[[1,3]])

		am[:a].mask.mask[[2,4]] .= false
		@test matcharrays(yarason(an, am, e) , [Dict(:a => "ca",:b => "sa"), Dict(:a => absent,:b => "sb"), Dict(:a => "cc",:b => "sc"), Dict(:a => absent,:b => "sd"), Dict(:a => "__UNKNOWN__",:b => "se")])
		am[:b].mask.mask[[2,4]] .= false
		@test matcharrays(yarason(an, am, e) , [Dict(:a => "ca",:b => "sa"), Dict(:a => absent,:b => absent), Dict(:a => "cc",:b => "sc"), Dict(:a => absent,:b => absent), Dict(:a => "__UNKNOWN__",:b => "se")])
		am[:a].mask.mask .= true
		@test matcharrays(yarason(an, am, e) , [Dict(:a => "ca",:b => "sa"), Dict(:a => "cb",:b => absent), Dict(:a => "cc",:b => "sc"), Dict(:a => "cd",:b => absent), Dict(:a => "__UNKNOWN__",:b => "se")])

		y = yarason(an, am, e)
		y = OR(y[1:2])
		@test !match(an[3:end], y, e)
		@test match(an[1:2], y, e)

		am[:a].mask.mask .= true
		am[:b].mask.mask .= true
		am[:a].mask.mask[[2,4]] .= false
		@test matcharrays(yarason(an, am, e, [true, false, false, true, false]) ,[Dict(:a => "ca",:b => "sa"), Dict(:a => absent,:b => "sd")])
		am[:b].mask.mask[[2,4]] .= false
		@test matcharrays(yarason(an, am, e, [true, false, false, true, false]) , [Dict(:a => "ca",:b => "sa"), Dict(:a => absent,:b => absent)])
		@test matcharrays(yarason(an, am, e, [true, false, true, false, false]) ,  [Dict(:a => "ca",:b => "sa"), Dict(:a => "cc",:b => "sc")])

		# Let's test the usecase, when we join scalar values and matrix to one big matrix
		e = ExtractDict(
			Dict(:n1 => JsonGrinder.ExtractScalar(Float32, 0, 1), 
				:n2 => JsonGrinder.ExtractScalar(Float32, 2, 1), 
				:n3 => JsonGrinder.ExtractScalar(Float32, 0//1, 1//3), 
				)
			, Dict(:a => ExtractCategorical(["a","b"]), )
			)

		s = [Dict(:n1 => "1",:n2 => 2,:n3 => 3,:a => "a"),
		 	 Dict(:n1 => "-1",:n2 => -2,:n3 => -3,:a => "b")]

		an = reduce(catobs,e.(s))
		am = Mask(an, d -> rand(d))
		y = yarason(an, am, e)
		@test matcharrays(y, [Dict(:a => "a",:n3 => 3.0f0,:n1 => 1.0f0,:n2 => 2.0f0), Dict(:a => "b",:n3 => -3.0f0,:n1 => -1.0f0,:n2 => -2.0f0)])
		@test Base.match(an, y[1], e)	
		@test Base.match(an, y[2], e)	
		@test !Base.match(an[1], y[2], e)	
		@test !Base.match(an[2], y[1], e)	
		am[:scalars].mask.mask .= [false,true,false]
		y = yarason(an, am, e)
		@test matcharrays(y, [Dict(:a => "a",:n3 => absent,:n1 => 1.0f0,:n2 => absent), Dict(:a => "b",:n3 => absent,:n1 => -1.0f0,:n2 => absent)])
		@test Base.match(an, y[1], e)	
		@test Base.match(an, y[2], e)	
		@test !Base.match(an[1], y[2], e)	
		@test !Base.match(an[2], y[1], e)	
	end
	
	@testset "logicaland" begin
		@test logicaland(1, absent) == 1
		@test logicaland(absent, 1) == 1
		@test isabsent(logicaland(absent, absent))
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

		y = yarason(an, am, e)
		@test matcharrays(y , s)
		@test match(an, y, e)
		@test match(an, y[1:1], e)
		@test !match(an[1], y, e)
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

		y = yarason(an, am, e)
		@test matcharrays(y, [[Dict("ka" => ["a1", "a2", "a3"])], [Dict("ka" => ["a1", "a2"]), Dict("kc" => ["c1"])]])
		@test match(an, y[1], e)
		@test !match(an[1], y[2], e)
		@test !match(an[2], y[1], e)
		@test !match(an, Dict("kc" => ["a1", "a2", "a3"]), e)
		@test match(an, Dict(absent => ["a1", "a2", "a3"]), e)
		@test match(an, Dict("kc" => absent), e)

		@test matcharrays(yarason(an, am, e, [true, false]) , [[Dict("ka" => ["a1", "a2", "a3"])]])
		am.mask.mask .= [false, true, true]
		@test matcharrays(yarason(an, am, e), [[], [Dict("ka" => ["a1", "a2"]), Dict("kc" => ["c1"])]])
		am.mask.mask .= [true, false, true]
		@test matcharrays(yarason(an, am, e),  [[Dict("ka" => ["a1", "a2", "a3"])], [Dict("kc" => ["c1"])]])
		am.mask.mask .= true
		am.mask.mask .= [true, false, true]
		am.child[:key].mask.mask .=[false, true, true]
		@test matcharrays(yarason(an, am, e),  [[Dict(absent => ["a1", "a2", "a3"])], [Dict("kc" => ["c1"])]])

		#we need to ensure that we can match absets!!!!

	end 

	@testset "removing absent" begin
		@test removeabsent([1,2,3,absent]) == [1,2,3]
		@test removeabsent([[1],[2],[3],absent]) == [[1],[2],[3]]
		@test removeabsent([[1],[2],[absent],absent]) == [[1],[2]]
		@test removeabsent([[1],"a",[absent],absent]) == [[1],"a"]
		@test matcharrays(removeabsent(Dict(:a => [[1],"a",[absent],absent], :b => absent, :c => [absent])), Dict((:a => [[1], "a"])))
	end
end
