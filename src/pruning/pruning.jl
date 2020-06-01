include("utils.jl")
include("greedy.jl")
include("importantfirst.jl")
include("breadthfirst.jl")


function prune!(f, ms, scorefun, method)
	if method == :greedy
		@timeit to "greedy" ExplainMill.greedy!(f, ms, scorefun)
	elseif method == :abstemious
		@timeit to "greedy" ExplainMill.greedy!(f, ms, scorefun; rev = false)
	elseif method == :importantfirst
		@timeit to "importantfirst" ExplainMill.importantfirst!(f, ms, scorefun)
	elseif method == :oscilatingimportantfirst
		@timeit to "oscilatingimportantfirst" ExplainMill.importantfirst!(f, ms, scorefun; oscilate = true)
	elseif method == :importantlast
		@timeit to "importantfirst" ExplainMill.importantfirst!(f, ms, scorefun; rev = false)
	elseif method == :breadthfirst
		@timeit to "breadthfirst" ExplainMill.breadthfirst!(f, ms, scorefun)
	elseif method == :breadthfirst2
		@timeit to "breadthfirst2" ExplainMill.breadthfirst2!(f, ms, scorefun)
	elseif method == :oscilatingbreadthfirst
		@timeit to "oscilatebreadthfirst2" ExplainMill.breadthfirst2!(f, ms, scorefun, oscilate = true)
	elseif method == :oscilatingsfs
		@timeit to "oscilatingsfs" ExplainMill.sequentialfs!(f, ms, scorefun, oscilate = true)
	elseif method == :sfs
		@timeit to "sfs" ExplainMill.sequentialfs!(f, ms, scorefun)
	end
end