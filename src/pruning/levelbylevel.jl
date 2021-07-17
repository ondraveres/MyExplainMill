"""
	add_participation(mk::AbstractStructureMask)

	decorate masks with `ParticipationTracker` if they do not 
	support participation off the shelf. If they do, do nothing.
"""
function add_participation(mk)
	s = true 
	foreach_mask(mk) do m, _
		s = s & support_participation(m)
	end
	s && return(mk)

	mapmask(mk) do m, _ 
		(m isa AbstractNoMask) && return(m)
		support_participation(m) ? m : ParticipationTracker(m)
	end
end


function levelbylevelsearch!(f, mk::AbstractStructureMask; levelsearch! = flatsearch!, fine_tuning::Bool = false, random_removal::Bool = true)
	_levelbylevelsearch!(f, add_participation(mk), levelsearch!, fine_tuning, random_removal)
end

function _levelbylevelsearch!(f, mk::AbstractStructureMask, levelsearch!, fine_tuning, random_removal)
	mk = add_participation(mk)
	all_masks = collect_masks_with_levels(mk)
	if isempty(all_masks) 
		@warn "Cannot explain empty samples"
		return()
	end

	full_flat = FlatView(map(first, all_masks))
	full_flat .= true
	for level in 1:maximum(map(x -> x.second, all_masks))
		level_masks = filter(m -> m.second == level, all_masks)
		isempty(level_masks) && continue
		level_flat = FlatView(map(first, level_masks))
		updateparticipation!(mk)
		@debug "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		levelsearch!(f, level_flat; participateonly = true, random_removal = random_removal, fine_tuning = fine_tuning)
	end

	random_removal && randomremoval!(f, full_flat)
	fine_tuning && finetune!(f, full_flat, 5)
	used = useditems(full_flat)

	# ensure that non-participating are set to false

	@debug "Explanation uses $(length(used)) features out of $(length(full_flat))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end

function levelbylevelsearch!(ms::AbstractStructureMask, model::AbstractMillModel, ds::AbstractNode, threshold, i, scorefun; fine_tuning::Bool = false, random_removal::Bool = true)
	# sort all explainable masks by depth and types
	parents = parent_structure(ms)
	parents = filter(x -> !isa(x.first, AbstractNoMask), parents)
	masks = map(x -> x.first, parents)

	#get rid of masks, which does not have any explainable item
	dp = map(x -> x.second, parents)

	max_depth = maximum(dp)
	fullmask = FlatView(masks)
	fill!(fullmask, true)
	f = () -> sum(min.(ExplainMill.confidencegap(ds -> softmax(model(ds)), ds[ms], i) .- threshold, 0))
	for j in 1:max_depth
		levelmasks = masks[dp .== j]
		isempty(levelmasks) && continue

		fv, significance = prepare_level!(levelmasks, ms, parents, scorefun)
		parmodel, pards, parms, changed = Mill.partialeval(model, ds, ms, levelmasks)

		@debug "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		@debug "output on the full sample before flat search $(f())"
		parf = () -> sum(min.(ExplainMill.confidencegap(x -> softmax(parmodel(x)), pards[parms], i) .- threshold, 0))
		flatsearch!(parf, fv, significance; participateonly = true, random_removal = random_removal, fine_tuning = fine_tuning)
		@debug "output on the full sample after flat search $(f())"
	end

	random_removal && randomremoval!(f, fullmask)
	# fine_tuning && finetune!(f, fullmask, 5)
	used = useditems(fullmask)
	@debug "Explanation uses $(length(used)) features out of $(length(fullmask))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end

function levelbylevelsfs!(f, mk::AbstractStructureMask; kwargs...)
	levelbylevelsearch!(f, mk; levelsearch! = flatsfs!, kwargs...)
end



