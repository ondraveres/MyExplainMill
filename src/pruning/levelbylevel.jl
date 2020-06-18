function prepare_level!(m, ms, parents, scorefun)
	fv = FlatView(firstparents(m, parents))
	updateparticipation!(ms)
	significance = map(scorefun, fv)
	fv, significance
end

function levelbylevelsearch!(f, ms::AbstractExplainMask, scorefun; fine_tuning::Bool = false, random_removal::Bool = true)
	# sort all explainable masks by depth and types
	parents = parent_structure(ms)
	masks = map(x -> x.first, parents)

	#get rid of masks, which does not have any explainable item
	masks = filter(x -> !isa(x, AbstractNoMask), masks)

	dp = map(masks) do x
		length(allparents(masks, parents, idofnode(x, parents)))
	end

	max_depth = maximum(dp)
	fullmask = FlatView(ms)
	fill!(fullmask, true)
	for j in 1:max_depth
		m = masks[dp .== j]
		isempty(m) && continue
		fv, significance = prepare_level!(m, ms, parents, scorefun)
		@info "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		flatsearch!(f, fv, significance; participateonly = true, random_removal = random_removal, fine_tuning = fine_tuning)
	end

	random_removal && randomremoval!(f, fullmask)
	fine_tuning && finetune!(f, fullmask, 5)
	used = useditems(fullmask)
	@info "Explanation uses $(length(used)) features out of $(length(fullmask))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end

function levelbylevelsfs!(f, ms::AbstractExplainMask, scorefun; fine_tuning::Bool = false, random_removal::Bool = false)
	# sort all explainable masks by depth and types
	parents = parent_structure(ms)
	masks = map(x -> x.first, parents)

	#get rid of masks, which does not have any explainable item
	masks = filter(x -> !isa(x, AbstractNoMask), masks)

	dp = map(masks) do x
		length(allparents(masks, parents, idofnode(x, parents)))
	end

	max_depth = maximum(dp)
	fullmask = FlatView(ms)
	fill!(fullmask, true)
	for j in 1:max_depth
		m = masks[dp .== j]
		isempty(m) && continue
		fv, significance = prepare_level!(m, ms, parents, scorefun)
		@info "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		flatsfs!(f, fv, random_removal = random_removal, fine_tuning = fine_tuning)
		@info "$(f()) uses $(length(useditems(fv))) with output $(f())"
	end

	used = useditems(fullmask)
	@info "Explanation uses $(length(used)) features out of $(length(fullmask))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end
