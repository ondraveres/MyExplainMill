"""
	explain(e, ds::AbstractNode, model::AbstractMillModel, i, n, clustering = ExplainMill._nocluster; threshold = nothing, pruning_method=:LbyL_HAdd, gap = 0.9f0)

	explain mask of a sample(s) ds, such that the confidencegap of the explanation is either above `threshold` (if set)
	or above the `gap*confidencegap` of the full sample(s) with `gap` being the `0.9` by default. 
	i is the index of the class which we are explaining and `n` is the number of repetitions / gradient 
	iterations in the calculation of stats.
"""
function explain(e, ds::AbstractNode, model::AbstractMillModel, i, n, clustering = ExplainMill._nocluster; threshold = nothing, pruning_method=:LbyL_HAdd, gap = 0.9f0)
	ms = ExplainMill.stats(e, ds, model, i, n)
	soft_model = ds -> softmax(model(ds))
	f = if nobs(ds) == 1
			threshold = (threshold == nothing) ? 0.9*ExplainMill.confidencegap1(soft_model, ds, i) : threshold
			() -> ExplainMill.confidencegap1(soft_model, ds[ms], i) - threshold
		else
			threshold = (threshold == nothing) ? 0.9.*ExplainMill.confidencegap(soft_model, ds, i) : threshold
			() -> sum(min.(ExplainMill.confidencegap(soft_model, ds[ms], i) .- threshold, 0))	
		end
	@timeit to "pruning" prune!(f, ms, x -> ExplainMill.scorefun(e, x), pruning_method)
	ms
end

function explain(e::GradExplainer, ds::AbstractNode, model::AbstractMillModel, i, n, clustering = ExplainMill._nocluster; threshold = nothing, pruning_method=:LbyL_HAdd, gap = 0.9f0)
	ms = ExplainMill.stats(e, ds, model, i, n)
	soft_model = ds -> softmax(model(ds))
	f = if nobs(ds) == 1
			threshold = (threshold == nothing) ? 0.9*ExplainMill.confidencegap1(soft_model, ds, i) : threshold
			() -> ExplainMill.confidencegap1(soft_model, ds[ms], i) - threshold
		else
			threshold = (threshold == nothing) ? 0.9.*ExplainMill.confidencegap(soft_model, ds, i) : threshold
			() -> sum(min.(ExplainMill.confidencegap(soft_model, ds[ms], i) .- threshold, 0))	
		end
	@timeit to "pruning" prune!(f, ms, x -> ExplainMill.scorefun(e, x), pruning_method)
	ms
end
