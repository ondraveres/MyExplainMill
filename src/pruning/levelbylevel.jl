"""
	add_participation(mk::AbstractStructureMask)

	decorate masks with `ParticipationTracker` if they do not 
	support participation off the shelf. If they do, do nothing.
"""
function add_participation(mk)
	support_participation(mk) && return(mk)
	mapmask(mk) do m, _ 
		(m isa AbstractNoMask) && return(m)
		support_participation(m) ? m : ParticipationTracker(m)
	end
end

"""

	support_participation(mk)

	true if all masks in `mk` supports tracking participation, which is 
	needed for effective `levelbylevelsearch`. Note that this is very complicated 
	with
"""
function support_participation(mk)
	s = true 
	foreach_mask(mk) do m, _
		s = s & support_participation(m)
	end
	s
end

"""
	levelbylevelsearch!(f, mk::AbstractStructureMask; levelsearch! = flatsearch!, fine_tuning::Bool = false, random_removal::Bool = true)
	levelbylevelsearch!(f, model::AbstractMillModel, ds::AbstractNode, mk::AbstractStructureMask; levelsearch! = flatsearch!, fine_tuning::Bool = false, random_removal::Bool = true)

	removes excess of items from `mk` such that `f` is above zero. 
	`f` is a function without parameters closing over `model`, `sample`, 
	and `mask`. 

	The second version is a more optimal, as it partially evaluates samples. 
	For example if we are explaining `BagNode`, we do not need to constantly 
	reevaluate its childs, as they are not influenced by the `BagMask` 
	corresponding to the `BagNode` in question. I expect this to speed up 
	the pruning of large samples, but it is a bit fragile. In this case,
	function `f` has to accept `f(model, sample, mask)`, since all three 
	are instantiated per level.



Example:
```julia
mk = stats(e, ds, model)
o = softmax(model(ds).data)[:]
τ = 0.9 * maximum(o) 
class = argmax(softmax(model(ds).data)[:])
f = () -> softmax(model(ds[mk]).data)[class] - τ
ExplainMill.levelbylevelsearch!(f, mk)
```

Alternatively for the partial evaluation
```
f = (model, ds, mk) -> softmax(model(ds[mk]).data)[class] - τ
ExplainMill.levelbylevelsearch!(f, model, ds, mk, random_removal = true)
```
"""	
function levelbylevelsearch!(f, mk::AbstractStructureMask; levelsearch! = flatsearch!, fine_tuning::Bool = false, random_removal::Bool = true)
	!support_participation(mk) && error("Level by level can be used only with masks supporting tracking of participation")
	all_masks = collect_masks_with_levels(mk)
	if isempty(all_masks) 
		@warn "Cannot explain empty samples"
		return()
	end

	full_flat = FlatView(all_masks)
	full_flat .= true
	for level in 1:maximum(map(x -> x.second, all_masks))
		updateparticipation!(mk)
		full_flat .= copy2vec(full_flat) .& participate(full_flat)
		level_masks = filter(m -> m.second == level, all_masks)
		isempty(level_masks) && continue
		level_flat = FlatView(level_masks)
		@debug "depth: $level length of mask: $(length(level_flat)) participating: $(sum(participate(level_flat)))"
		levelsearch!(f, level_flat; participateonly = true, random_removal , fine_tuning)
	end

	random_removal && randomremoval!(f, full_flat)
	fine_tuning && finetune!(f, full_flat, 5)
	used = useditems(full_flat)

	# ensure that non-participating are set to false
	full_flat .= copy2vec(full_flat) .& participate(full_flat)

	@debug "Explanation uses $(length(used)) features out of $(length(full_flat))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end


function levelbylevelsearch!(f, model::AbstractMillModel, ds::AbstractNode, mk::AbstractStructureMask; levelsearch! = flatsearch!, fine_tuning::Bool = false, random_removal::Bool = true)
	!support_participation(mk) && error("Level by level can be used only with masks supporting tracking of participation")
	all_masks = collect_masks_with_levels(mk)
	if isempty(all_masks) 
		@warn "Cannot explain empty samples"
		return()
	end

	full_flat = FlatView(all_masks)
	full_flat .= true
	_f = () -> f(model, ds, mk)
	for level in 1:maximum(map(x -> x.second, all_masks))
		updateparticipation!(mk)
		full_flat .= copy2vec(full_flat) .& participate(full_flat)
		level_masks = filter(m -> m.second == level, all_masks)
		isempty(level_masks) && continue

		modelₗ, dsₗ, mkₗ = partialeval(model, ds, mk, map(first, level_masks))
		# @debug "depth: $level length of mask: $(length(level_flat)) participating: $(sum(participate(level_flat)))"
		levelsearch!(() -> f(modelₗ, dsₗ, mkₗ), FlatView(level_masks); participateonly = true, random_removal, fine_tuning)
		@assert f(modelₗ, dsₗ, mkₗ) ≈ _f()
	end

	random_removal && randomremoval!(_f, full_flat)
	fine_tuning && finetune!(_f, full_flat, 5)
	used = useditems(full_flat)
	# ensure that non-participating are set to false
	full_flat .= copy2vec(full_flat) .& participate(full_flat)
	@debug "Explanation uses $(length(used)) features out of $(length(full_flat))"
	_f() < 0 && @error "output of explaination is $(_f()) and should be zero"
end

function levelbylevelsfs!(f, mk::AbstractStructureMask; kwargs...)
	levelbylevelsearch!(f, mk; levelsearch! = flatsfs!, kwargs...)
end

function levelbylevelsfs!(f, model::AbstractMillModel, ds::AbstractNode, mk::AbstractStructureMask; kwargs...)
	levelbylevelsearch!(f, model, ds, mk; levelsearch! = flatsfs!, kwargs...)
end



