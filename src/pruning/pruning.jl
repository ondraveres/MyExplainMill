include("utils.jl")
include("flatsearch.jl")
include("levelbylevel.jl")
include("greedy_gradient.jl")

function pruning_methods() 
	[:Flat_HAdd, :Flat_HArr, :Flat_HArrft, :Flat_Gadd, :Flat_Garr, 
	:Flat_Garrft, :LbyL_HAdd, :LbyL_HArr, :LbyL_HArrft, :LbyL_Gadd, 
	:LbyL_Garr, :LbyL_Garrft, :LbyLo_HAdd, :LbyLo_HArr, :LbyLo_HArrft, 
	:LbyLo_Gadd, :LbyLo_Garr, :LbyLo_Garrft]
end

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
function prune!(f, mk::AbstractStructureMask, method::Symbol)
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

function prune!(f, model::Mill.AbstractMillModel, ds::Mill.AbstractMillNode, mk::AbstractStructureMask, method::Symbol)
	if method == :LbyLo_HAdd
		ExplainMill.levelbylevelsearch!(f, model, ds, mk, random_removal = false, fine_tuning = false)
	elseif method == :LbyLo_HArr
		ExplainMill.levelbylevelsearch!(f, model, ds, mk, random_removal = true, fine_tuning = false)
	elseif method == :LbyLo_HArrft
		ExplainMill.levelbylevelsearch!(f, model, ds, mk, random_removal = true, fine_tuning = true)
	elseif method == :LbyLo_Gadd
		ExplainMill.levelbylevelsfs!(f, model, ds, mk)
	elseif method == :LbyLo_Garr
		ExplainMill.levelbylevelsfs!(f, model, ds, mk, random_removal = true)
	elseif method == :LbyLo_Garrft
		ExplainMill.levelbylevelsfs!(f, model, ds, mk, random_removal = true, fine_tuning = true)
	else
		error("Uknown pruning method $(method). Possible values (Flat_HArr, Flat_HArrft, Flat_Gadd, Flat_Garr, Flat_Garrft, LbyL_HAdd, LbyL_HArr, LbyL_HArrft, Flat_Gadd, Flat_Garr, Flat_Garrft)")
	end
end


function prune!(mk::AbstractStructureMask, model::AbstractMillModel, ds::AbstractMillNode, fₚ, method)
    mkp = add_participation(mk)

    if method ∈ [:LbyLo_HAdd, :LbyLo_HArr, :LbyLo_HArrft, :LbyLo_Gadd, :LbyLo_Garr, :LbyLo_Garrft]
        return prune!((model, ds, mk) -> fₚ(model(ds[mk]).data), model, ds, mkp, method)
    end
    # a fallback
    return prune!(() -> fₚ(model(ds[mkp]).data), mkp, method)
end