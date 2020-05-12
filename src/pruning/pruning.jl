include("utils.jl")
include("greedy.jl")
include("importantfirst.jl")
include("breadthfirst.jl")


function prune!(f, ms, scorefun, method)
	if method == :greedy
		@timeit to "greedy" ExplainMill.greedy!(f, ms, scorefun)
	elseif method == :importantfirst
		@timeit to "importantfirst" ExplainMill.importantfirst!(f, ms, scorefun)
	elseif method == :breadthfirst
		@timeit to "breadthfirst" ExplainMill.breadthfirst!(f, ms, scorefun)
	end
	elseif method == :breadthfirst2
		@timeit to "breadthfirst2" ExplainMill.breadthfirst2!(f, ms, scorefun)
	end
end