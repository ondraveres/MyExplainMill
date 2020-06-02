function prepare_breadthfirst!(m, ms, parents, scorefun)
	fv = FlatView(firstparents(m, parents))
	updateparticipation!(ms)
	significance = map(scorefun, fv)
	fv, significance
end

function breadthfirst!(f, ms::AbstractExplainMask, scorefun)
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
	#Let's first learn the BagNodes
	for j in 1:max_depth
		m = filter(x -> isa(x, BagMask), masks[dp .== j])
		isempty(m) && continue
		fv, significance = prepare_breadthfirst!(m, ms, parents, scorefun)
		@info "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		importantfirst!(f, fv, significance; participateonly = true)
	end

	# then, we switch to leaves and we will explain only participating leaves
	m = filter(x -> !isa(x, BagMask), masks)
	if !isempty(m)
		fv, significance = prepare_breadthfirst!(m, ms, parents, scorefun)
		foreach(x -> x.mask.mask .= x.mask.participate, m)
		used = sortindices(useditems(fv), significance, rev = false)
		@assert all(fv[used])
		removeexcess!(f, fv, used)
	end

	used = useditems(fullmask)
	@info "Explanation uses $(length(used)) features out of $(length(fullmask))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end

function breadthfirst2!(f, ms::AbstractExplainMask, scorefun; oscilate::Bool = false, random_removal::Bool = false)
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
		fv, significance = prepare_breadthfirst!(m, ms, parents, scorefun)
		@info "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		importantfirst!(f, fv, significance; participateonly = true, random_removal = random_removal)
		oscilate && oscilate!(f, fv)
	end

	used = useditems(fullmask)
	@info "Explanation uses $(length(used)) features out of $(length(fullmask))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end

function sequentialfs!(f, ms::AbstractExplainMask, scorefun; oscilate::Bool = false, random_removal::Bool = false)
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
		fv, significance = prepare_breadthfirst!(m, ms, parents, scorefun)
		@info "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		sfs!(f, fv, random_removal = random_removal)
		oscilate && oscilate!(f, fv)
		@info "$(f()) uses $(length(useditems(fv))) with output $(f())"
	end

	used = useditems(fullmask)
	@info "Explanation uses $(length(used)) features out of $(length(fullmask))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end

function sfs!(f, ms::AbstractExplainMask, scorefun; oscilate::Bool = false, random_removal::Bool = false)
	# sort all explainable masks by depth and types
	parents = parent_structure(ms)
	masks = map(x -> x.first, parents)

	#get rid of masks, which does not have any explainable item
	masks = filter(x -> !isa(x, AbstractNoMask), masks)
	fv = FlatView(ms)
	@info "sfs: length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
	sfs!(f, fv, random_removal = random_removal)
	oscilate && oscilate!(f, fv)
	@info "$(f()) uses $(length(useditems(fv))) with output $(f())"
end