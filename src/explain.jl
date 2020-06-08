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
