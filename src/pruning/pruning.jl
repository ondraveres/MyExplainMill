include("utils.jl")
include("flatsearch.jl")
include("levelbylevel.jl")
include("gradient_submodular.jl")

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
function prune!(f, mk, method)
	if method == :Flat_HAdd
		ExplainMill.flatsearch!(f, mk, random_removal = false, fine_tuning = false)
	elseif method == :Flat_HArr
		ExplainMill.flatsearch!(f, mk, random_removal = true, fine_tuning = false)
	elseif method == :Flat_HArrft
		ExplainMill.flatsearch!(f, mk, random_removal = true, fine_tuning = true)
	elseif method == :Flat_Gadd
		ExplainMill.flatsfs!(f, mk)
	elseif method == :Flat_Garr
		ExplainMill.flatsfs!(f, mk, random_removal = true)
	elseif method == :Flat_Garrft
		ExplainMill.flatsfs!(f, mk, random_removal = true, fine_tuning = true)
	elseif method == :LbyL_HAdd
		ExplainMill.levelbylevelsearch!(f, mk, random_removal = false, fine_tuning = false)
	elseif method == :LbyL_HArr
		ExplainMill.levelbylevelsearch!(f, mk, random_removal = true, fine_tuning = false)
	elseif method == :LbyL_HArrft
		ExplainMill.levelbylevelsearch!(f, mk, random_removal = true, fine_tuning = true)
	elseif method == :LbyL_Gadd
		ExplainMill.levelbylevelsfs!(f, mk)
	elseif method == :LbyL_Garr
		ExplainMill.levelbylevelsfs!(f, mk, random_removal = true)
	elseif method == :LbyL_Garrft
		ExplainMill.levelbylevelsfs!(f, mk, random_removal = true, fine_tuning = true)
	else
		error("Uknown pruning method $(method). Possible values (Flat_HArr, Flat_HArrft, Flat_Gadd, Flat_Garr, Flat_Garrft, LbyL_HAdd, LbyL_HArr, LbyL_HArrft, Flat_Gadd, Flat_Garr, Flat_Garrft)")
	end
end

function prune!(mk::AbstractStructureMask, model::AbstractMillModel, ds::AbstractNode, i, thresholds, method)
    soft_model(x) = softmax(model(x))

    if nobs(ds) == 1 ||
        method ∈ [:Flat_HAdd, :Flat_HArr, :Flat_HArrft, :Flat_GAdd, :Flat_GArr, :Flat_GArrft, :LbyL_Gadd, :LbyL_Garr, :LbyL_Garrft]
        f = () -> sum(min.(ExplainMill.confidencegap(soft_model, ds[mk], i) .- thresholds, 0))
        return prune!(f, mk, method)
    end

    fine_tuning = method == :LbyL_HArrft
    random_removal = method ∈ [:LbyL_HArr, :LbyL_HArrft]
    levelbylevelsearch!(mk, model, ds, thresholds, i; fine_tuning = fine_tuning, random_removal = random_removal)
end
