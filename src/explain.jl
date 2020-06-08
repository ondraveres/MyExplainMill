function explain(e, ds::AbstractNode, model::AbstractMillModel, i, n, clustering = ExplainMill._nocluster; threshold = 0.1, pruning_method=:LbyL_HAdd)
	ms = ExplainMill.stats(e, ds, model, i, n)
	soft_model = ds -> softmax(model(ds))
	f = () -> ExplainMill.confidencegap(soft_model, ds[ms], i) .- threshold
	if nobs(ds) > 1
		f = () -> sum(min.(ExplainMill.confidencegap(soft_model, ds[ms], i) .- threshold, 0))	
	end
	@timeit to "pruning" prune!(f, ms, x -> ExplainMill.scorefun(e, x), pruning_method)
	ms
end

function explain(e::GradExplainer, ds::AbstractNode, model::AbstractMillModel, i, n, clustering = ExplainMill._nocluster; threshold = 0.1, pruning_method=:LbyL_HAdd)
	ms = ExplainMill.stats(e, ds, model, i, n)
	soft_model = ds -> softmax(model(ds))
	f = () -> ExplainMill.confidencegap(soft_model, ds[ms], i) .- threshold
	if nobs(ds) > 1
		f = () -> sum(min.(ExplainMill.confidencegap(soft_model, ds[ms], i) .- threshold, 0))	
	end
	@timeit to "pruning" prune!(f, ms, x -> ExplainMill.scorefun(e, x), pruning_method)
	ms
end


####
# A hacky POC, where DAF statistics are calculated layer by layer. 
####
function updatestats!(e::DafExplainer, fv::FlatView, ds, ms, soft_model, i, n)
	f = e.hard ? () -> output(soft_model(ds[ms]))[i,:] : () -> output(soft_model(ds, ms))[i,:]
	for _ in 1:n
		map(m -> sample!(m.mask), fv) 
		o = @timeit to "evaluate" f()
		map(m -> Duff.update!(m.mask, o), fv) 
	end
end

function explaindepthwise(e, ds::AbstractNode, model::AbstractMillModel, i, n, clustering = ExplainMill._nocluster; threshold = 0.1)
	ms = ExplainMill.stats(e, ds, model, i, 0)
	soft_model = ds -> softmax(model(ds))
	f = () -> sum(min.(ExplainMill.confidencegap(soft_model, ds[ms], i) .- threshold, 0))	

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
		updateparticipation!(ms)
		m = filter(x -> isa(x, BagMask), masks[dp .== j])
		isempty(m) && continue
		updatestats!(e, FlatView(firstparents(m, parents)), ds, ms, soft_model, i, n)
		fv, significance = prepare_breadthfirst!(m, ms, parents, x -> scorefun(e, x))
		@info "depth: $(j) length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
		importantfirst!(f, fv, significance; participateonly = true)
	end
	updateparticipation!(ms)

	#then, we switch to leaves and we will explain only participating leaves
	m = filter(x -> !isa(x, BagMask), masks)
	if !isempty(m)
		fv, significance = prepare_breadthfirst!(m, ms, parents, x -> scorefun(e, x))
		foreach(x -> x.mask.mask .= x.mask.participate, m)
		used = sortindices(useditems(fv), significance, rev = false)
		@assert all(fv[used])
		removeexcess!(f, fv, used)
	end

	used = useditems(fullmask)
	@info "Explanation uses $(length(used)) features out of $(length(fullmask))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
	ms
end

