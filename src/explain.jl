"""
	scalar(x) 

	scalar from 1-by-1 matrix, vector with a single item, or an ArrayNode
"""
@inline function scalar(x::Matrix) 
	@assert size(x, 1) == 1
	@assert size(x, 2) == 1
	return(x[1])
end
@inline function scalar(x::Vector) 
	@assert size(x, 1) == 1
	return(x[1])
end
@inline scalar(x::Mill.ArrayNode) = scalar(x.data)
@inline scalar(x) = x


"""
	function dafstats(ds, model, n=10000)

	Shapley values of individual items of a sample `ds` in the model `model` estimated from `n` trials
"""
function dafstats(ds, model, n=10000)
	daf = Duff.Daf(ds);
	for i in 1:n
		dss, mask = sample(daf, ds)
		v = scalar(model(dss))
		Duff.update!(daf, mask, v)
	end
	return(daf)
end


"""
	explain(ds, model;  n = 10000, method = :uselessfirst, threshold = 0.5, verbose = false)

	Explain sample `ds` by removing its items such that output of `model(ds)` is above `threshold`.
	`method` controls the order in which the items are subjected to iterative removal. 
	:uselessfirst means that samples with low importance are removed first
"""
function explain(ds, model;  n = 10000, method = :uselessfirst, threshold = 0.5, verbose = false)
	if scalar(model(ds)) < threshold
		@info "stopped explanation as the output is below threshold"
		return(ds)
	end
	daf = dafstats(ds, model, n)
	mask, dafs = masks_and_stats(daf)
	if method == :uselessfirst
		return(uselessfirst(ds, model, mask, dafs, threshold, verbose))
	else
		@error "unknown pruning method $(method)"
	end
end


"""
	uselessfirst(ds, model, mask, dafs, threshold, verbose)

	Removes items from a sample `ds` such that output of `model(ds)` is above `threshold`.
	Removing starts with items that according to Shapley values do not contribute to the output
"""
function uselessfirst(ds, model, mask, dafs, threshold, verbose)
	catmask = CatView(tuple([d.m for d in dafs]...))
	pvalue = CatView(tuple([Duff.UnequalVarianceTTest(d.d) for d in dafs]...))

	verbose && println("model output: ",round(scalar(model(ds)), digits = 3))
	for i in sortperm(pvalue, rev = true)
		catmask[i] = false
		o = scalar(model(prune(ds, mask)))
		if o <= threshold
			catmask[i] = true
		end
		verbose && println(i,": p-value: ",round(pvalue[i], digits = 6),": model output: ",round(o, digits = 3))
	end
	return(prune(ds, mask))
end
