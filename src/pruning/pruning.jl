include("utils.jl")
include("flatsearch.jl")
include("levelbylevel.jl")


# :sfsrr => "LbyL-GArr",
# :oscilatingsfs => "LbyL-GAos",
# :sfs => "LbyL-GAdd",
# :flatsfsrr => "GArr",
# :flatsfsos => "GAos",
# :flatsfs => "GAdd",
# :greedy => "HAdd",
# :importantfirst => "HArr",
# :oscilatingimportantfirst => "HAos",
# :breadthfirst2 => "LbyL-HArr",
# :greedybreadthfirst => "LbyL-HAdd",
# :oscilatingbreadthfirst => "LbyL-HAos",
# :abstemious => "ab",
# :importantlast => "il",
function prune!(f, ms, scorefun, method)
	if method == :Flat_HAdd
		ExplainMill.flatsearch!(f, ms, scorefun, random_removal = false, fine_tuning = false)
	elseif method == :Flat_HArr
		ExplainMill.flatsearch!(f, ms, scorefun, random_removal = true, fine_tuning = false)
	elseif method == :Flat_HArrft
		ExplainMill.flatsearch!(f, ms, scorefun, random_removal = true, fine_tuning = true)
	elseif method == :Flat_Gadd
		ExplainMill.flatsfs!(f, ms, scorefun)
	elseif method == :Flat_Garr
		ExplainMill.flatsfs!(f, ms, scorefun, random_removal = true)
	elseif method == :Flat_Garrft
		ExplainMill.flatsfs!(f, ms, scorefun, random_removal = true, fine_tuning = true)
	elseif method == :LbyL_HAdd
		ExplainMill.levelbylevelsearch!(f, ms, scorefun, random_removal = false, fine_tuning = false)
	elseif method == :LbyL_HArr
		ExplainMill.levelbylevelsearch!(f, ms, scorefun, random_removal = true, fine_tuning = false)
	elseif method == :LbyL_HArrft
		ExplainMill.levelbylevelsearch!(f, ms, scorefun, random_removal = true, fine_tuning = true)
	elseif method == :Flat_Gadd
		ExplainMill.levelbylevelsfs!(f, ms, scorefun)
	elseif method == :Flat_Garr
		ExplainMill.levelbylevelsfs!(f, ms, scorefun, random_removal = true)
	elseif method == :Flat_Garrft
		ExplainMill.levelbylevelsfs!(f, ms, scorefun, random_removal = true, fine_tuning = true)
	else
		error("Uknown pruning method $(method)")
	end
end
