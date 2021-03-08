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
# :oscilatingbreadthfirst => "LbyL_HAos",
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
	elseif method == :LbyL_Gadd
		ExplainMill.levelbylevelsfs!(f, ms, scorefun)
	elseif method == :LbyL_Garr
		ExplainMill.levelbylevelsfs!(f, ms, scorefun, random_removal = true)
	elseif method == :LbyL_Garrft
		ExplainMill.levelbylevelsfs!(f, ms, scorefun, random_removal = true, fine_tuning = true)
	else
		error("Uknown pruning method $(method). Possible values (Flat_HArr, Flat_HArrft, Flat_Gadd, Flat_Garr, Flat_Garrft, LbyL_HAdd, LbyL_HArr, LbyL_HArrft, Flat_Gadd, Flat_Garr, Flat_Garrft)")
	end
end

function adjustthreshold(threshold::Nothing, gap, model, ds, i)
	m = x -> softmax(model(x))
	nobs(ds) == 1 && return(gap*ExplainMill.confidencegap1(m, ds, i))
	return(gap*ExplainMill.confidencegap(m, ds, i))
end
adjustthreshold(threshold, gap, model, ds, i) = threshold

function prune!(ms::AbstractExplainMask, model::AbstractMillModel, ds::AbstractNode, i, scorefun, threshold, method)
	soft_model(x) = softmax(model(x))
	if nobs(ds) == 1
		f = () -> ExplainMill.confidencegap1(soft_model, ds[ms], i) - threshold
		return(prune!(f, ms, scorefun, method))
	end

	if method ∈ [:Flat_HAdd, :Flat_HArr, :Flat_HArrft, :Flat_GAdd, :Flat_GArr, :Flat_GArrft, :LbyL_Gadd, :LbyL_Garr, :LbyL_Garrft]
		f = () -> sum(min.(ExplainMill.confidencegap(soft_model, ds[ms], i) .- threshold, 0))	
		return(prune!(f, ms, scorefun, method))
	end

	fine_tuning = method == :LbyL_HArrft
	random_removal = method ∈ [:LbyL_HArr, :LbyL_HArrft]
	levelbylevelsearch!(ms, model, ds, threshold, i, scorefun; fine_tuning = fine_tuning, random_removal = random_removal)
end
