function prepare_level!(m, ms, parents, scorefun)
	fv = FlatView(m)
	updateparticipation!(ms)
	significance = map(scorefun, fv)
	fv, significance
end

function levelbylevelsearch!(f, ms::AbstractStructureMask, scorefun; fine_tuning::Bool = false, random_removal::Bool = true)
	# sort all explainable masks by depth and types
	parents = parent_structure(ms)
	parents = filter(x -> !isa(x.first, AbstractNoMask), parents)
	masks = map(x -> x.first, parents)

	#get rid of masks, which does not have any explainable item

	if isempty(masks) 
		@warn "Cannot explain empty samples"
		return()
	end

	dp = map(x -> x.second, parents)
	max_depth = maximum(dp)
	fullmask = FlatView(masks)
	fill!(fullmask, true)
	for j in 1:max_depth
		m = masks[dp .== j]
		isempty(m) && continue
		fv, significance = prepare_level!(m, ms, parents, scorefun)
		@debug "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		flatsearch!(f, fv, significance; participateonly = true, random_removal = random_removal, fine_tuning = fine_tuning)
	end

	random_removal && randomremoval!(f, fullmask)
	# fine_tuning && finetune!(f, fullmask, 5)
	used = useditems(fullmask)
	@debug "Explanation uses $(length(used)) features out of $(length(fullmask))"
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

function levelbylevelsfs!(f, ms::AbstractStructureMask, scorefun; fine_tuning::Bool = false, random_removal::Bool = false)
	# sort all explainable masks by depth and types
	parents = parent_structure(ms)
	parents = filter(x -> !isa(x.first, AbstractNoMask), parents)
	masks = map(x -> x.first, parents)

	#get rid of masks, which does not have any explainable item
	masks = filter(x -> !isa(x, AbstractNoMask), masks)
	dp = map(x -> x.second, parents)

	max_depth = maximum(dp)
	fullmask = FlatView(masks)
	fill!(fullmask, true)
	for j in 1:max_depth
		m = masks[dp .== j]
		isempty(m) && continue
		fv, significance = prepare_level!(m, ms, parents, scorefun)
		@debug "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		flatsfs!(f, fv, random_removal = random_removal, fine_tuning = fine_tuning)
		@debug "$(f()) uses $(length(useditems(fv))) with output $(f())"
	end

	used = useditems(fullmask)
	@debug "Explanation uses $(length(used)) features out of $(length(fullmask))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end



