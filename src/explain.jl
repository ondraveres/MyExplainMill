"""
	explain(e, ds::AbstractNode, model::AbstractMillModel, i, n, clustering = ExplainMill._nocluster; threshold = nothing, pruning_method=:LbyL_HArr, gap = 0.9f0)

	explain mask of a sample(s) ds, such that the confidencegap of the explanation is either above `threshold` (if set)
	or above the `gap*confidencegap` of the full sample(s) with `gap` being the `0.9` by default.
	i is the index of the class which we are explaining and `n` is the number of repetitions / gradient
	iterations in the calculation of stats.
"""
function explain(e, ds::AbstractNode, model::AbstractMillModel, i::Int, clustering = ExplainMill._nocluster; threshold = nothing, pruning_method=:LbyL_HArr, gap = 0.9f0)
	threshold = adjustthreshold(threshold, gap, model, ds, i)
	minimum(ExplainMill.confidencegap(model, ds, i) .- threshold) < 0 && error("cannot explain samples with negative confidence")
	ms = ExplainMill.stats(e, ds, model, i, clustering)
	ExplainMill.prune!(ms, model, ds, i, x -> ExplainMill.scorefun(e, x), threshold, pruning_method)
	ms
end

#This cannot work, as f is not defined
# function explain(e::GradExplainer, ds::AbstractNode, model::AbstractMillModel, i::Int, clustering = ExplainMill._nocluster; threshold = nothing, pruning_method=:LbyL_HArr, gap = 0.9f0)
# 	minimum(ExplainMill.confidencegap(explaining_model, ds, i)) < 0 && error("cannot explain samples with negative confidence")
# 	ms = ExplainMill.stats(e, ds, model, i, clustering)
# 	prune!(f, ms, x -> ExplainMill.scorefun(e, x), pruning_method)
# 	ms
# end


function explain(e, ds::AbstractNode, model::AbstractMillModel, clustering = ExplainMill._nocluster; threshold = nothing, pruning_method=:LbyL_HArr, gap = 0.9f0)
	i = unique(Flux.onecold(softmax(model(ds).data)))
	length(i) > 1 && error("We can explain only data with the same output class.")
	i = only(i)
	explain(e, ds, model, i, clustering; threshold = threshold, pruning_method=pruning_method, gap = gap)
end

# function explain(e, ds::AbstractNode, negative_ds, model::AbstractMillModel, extractor::JsonGrinder.AbstractExtractor, clustering = ExplainMill._nocluster; threshold = nothing, pruning_method=:LbyL_HArr, gap = 0.9f0, max_repetitions = 10)
# 	i = unique(Flux.onecold(softmax(model(ds).data)))
# 	length(i) > 1 && error("We can explain only data with the same output class.")
# 	i = only(i)
# 	ms = ExplainMill.stats(e, ds, model, i, clustering)
# 	soft_model = ds -> softmax(model(ds))
# 	f = if nobs(ds) == 1
# 		threshold = (threshold == nothing) ? 0.9*ExplainMill.confidencegap1(soft_model, ds, i) : threshold
# 		() -> ExplainMill.confidencegap1(soft_model, ds[ms], i) - threshold
# 	else
# 		threshold = (threshold == nothing) ? 0.9.*ExplainMill.confidencegap(soft_model, ds, i) : threshold
# 		() -> sum(min.(ExplainMill.confidencegap(soft_model, ds[ms], i) .- threshold, 0))
# 	end

# 	#Let's try to explain without the negative samples
# 	prune!(f, ms, x -> ExplainMill.scorefun(e, x), pruning_method)
# 	yara = ExplainMill.e2boolean(ms, ds, extractor)
# 	fps = filter(dd -> match(yara, extractor, dd), negative_ds)
# 	@info "number of false positives  $(length(fps))"

# 	function optfun(fps)
# 		fval = f()
# 		fval < 0 && return(fval)
# 		yara = ExplainMill.e2boolean(ms, ds, extractor)
# 		y = any(map(dd -> match(yara, extractor, dd), fps))
# 		y ? typeof(fval)(-1) : fval
# 	end

# 	n = 0
# 	allfps = fps
# 	while !isempty(fps) && n < max_repetitions
# 		prune!(() -> optfun(allfps), ms, x -> ExplainMill.scorefun(e, x), pruning_method)
# 		fps = filter(ds -> match(yara, extractor, ds), negative_ds)
# 		allfps = union(fps, allfps)
# 		@info "number of false positives  $(length(fps))"
# 		n += 1

# 		if optfun(fps) < 0
# 			@info "Failed to find a feasible explanation"
# 			return(nothing)
# 		end
# 	end

# 	ms
# end

function explainy(e, ds::AbstractNode, negative_ds, model::AbstractMillModel, extractor::JsonGrinder.AbstractExtractor, clustering = ExplainMill._nocluster; threshold = nothing, pruning_method=:LbyL_HArr, gap = 0.9f0, max_repetitions = 10)
	i = unique(Flux.onecold(softmax(model(ds).data)))
	length(i) > 1 && error("We can explain only data with the same output class.")
	i = only(i)
	threshold = adjustthreshold(threshold, gap, model, ds, i)

	ms = ExplainMill.stats(e, ds, model, i, clustering)
	soft_model = ds -> softmax(model(ds))
	tps = [ds[i] for i in 1:nobs(ds)]

	function optfun(tps, fps)
		yara = ExplainMill.e2boolean(ms, ds, extractor)
		fval = mean(map(dd -> match(yara, extractor, dd), tps)) - 1
		isempty(fps) && return(fval)
		y = any(map(dd -> match(yara, extractor, dd), fps))
		y ? typeof(fval)(-1) : fval
	end

	n = 0
	allfps, fps = similar(negative_ds, 0), similar(negative_ds, 0)
	while true
		prune!(() -> optfun(tps, allfps), ms, x -> ExplainMill.scorefun(e, x), pruning_method)
		fps = filter(ds -> match(yara, extractor, ds), negative_ds)
		isempty(fps) && break
		allfps = union(fps, allfps)
		@info "number of false positives  $(length(fps))"
		n += 1

		if optfun(fps) < 0
			@info "Failed to find a feasible explanation"
			return(nothing)
		end
	end

	ms
end
