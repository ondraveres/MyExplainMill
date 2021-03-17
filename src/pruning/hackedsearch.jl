function hackedsearch!(f, ms::AbstractExplainMask, scorefun)
	# sort all explainable masks by depth and types
	parents = parent_structure(ms)
	masks = map(x -> x.first, parents)

	#get rid of masks, which does not have any explainable item
	masks = filter(x -> !isa(x, AbstractNoMask), masks)

	if isempty(masks) 
		@warn "Cannot explain empty samples"
		return()
	end

	dp = map(masks) do x
		length(allparents(masks, parents, idofnode(x, parents)))
	end

	max_depth = maximum(dp)
	fullmask = FlatView(ms)
	fill!(fullmask, true)
	f() < 0 && error("cannot explain when full sample has negative output")
	for j in 1:max_depth
		m = masks[dp .== j]
		isempty(m) && continue
		fv, significance = prepare_level!(m, ms, parents, scorefun)
		@debug "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		ExplainMill.branchandbound!(f, fv, significance)
		if f() < 0
			fill!(fv, true)
			@error "Failed to prune, returning full explanation"
			return false
		end
		break
	end

	used = useditems(fullmask)
	@debug "Explanation uses $(length(used)) features out of $(length(fullmask))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end

function hackedsearch!(ms::AbstractExplainMask, model::AbstractMillModel, ds::AbstractNode , scorefun)
	# sort all explainable masks by depth and types
	parents = parent_structure(ms)
	masks = map(x -> x.first, parents)

	#get rid of masks, which does not have any explainable item
	masks = filter(x -> !isa(x, AbstractNoMask), masks)

	if isempty(masks) 
		@warn "Cannot explain empty samples"
		return()
	end

	dp = map(masks) do x
		length(allparents(masks, parents, idofnode(x, parents)))
	end

	max_depth = maximum(dp)
	fullmask = FlatView(ms)
	fill!(fullmask, true)
	model(ds).data[1] < 0 && error("cannot explain when full sample has negative output")
	for j in 1:max_depth
		m = masks[dp .== j]
		isempty(m) && continue
		fv, significance = prepare_level!(m, ms, parents, scorefun)
		parmodel, pards, parms, changed = Mill.partialeval(model, ds, ms, m)

		@debug "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		f = () -> parmodel(pards[parms]).data[1]
		ExplainMill.branchandbound!(f, fv, significance)
		if f() < 0
			fill!(fv, true)
			@error "Failed to prune, returning full explanation"
			return false
		end
		break
	end

	used = useditems(fullmask)
	@debug "Explanation uses $(length(used)) features out of $(length(fullmask))"
	fval = model(ds[ms]).data[1]
	fval < 0 && @error "output of explaination is $(fval) and should be zero"
end
