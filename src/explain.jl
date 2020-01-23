Base.minimum(ds::ArrayNode) = minimum(ds.data)

function StatsBase.sample!(pruning_mask::AbstractExplainMask)
	mapmask(sample!, pruning_mask)
end

function Duff.update!(dafs::Vector, v::Mill.ArrayNode, pruning_mask)
	Duff.update!(dafs, v.data, pruning_mask)
end

function Duff.update!(dafs::Vector, v::AbstractArray{T}, pruning_mask) where{T<:Real}
	for i in 1:length(v)
		mapmask(pruning_mask) do m 
			participate(m) .= true
		end
		invalidate!(pruning_mask,setdiff(1:length(v), i))
		for d in dafs 
			Duff.update!(d, v[i])
		end
	end
end


"""
	function dafstats(ds, model, n=10000)

	Shapley values of individual items of a sample `ds` in the model `model` estimated from `n` trials
"""
function dafstats(ds, model, n=10000)
	pruning_mask = Mask(ds)
	dafs = []
	mapmask(pruning_mask) do m
		m != nothing && push!(dafs, m)
	end
	for i in 1:n
		@timeit to "sample!" sample!(pruning_mask)
		pruned_ds = @timeit to "prune" prune(ds, pruning_mask)
		o = @timeit to "evaluate" model(pruned_ds)
		@timeit to "update!" Duff.update!(dafs, o, pruning_mask)
	end
	return(dafs, pruning_mask)
end


"""
	explain(ds, model;  n = 10000, method = :uselessfirst, threshold = 0.5, verbose = false)

	Explain sample `ds` by removing its items such that output of `model(ds)` is above `threshold`.
	`method` controls the order in which the items are subjected to iterative removal. 
	:uselessfirst means that samples with low importance are removed first
"""
function explain(ds, model;  n = 10000, method = :uselessfirst, threshold = 0.5, verbose = false)
	if minimum(model(ds)) < threshold
		@info "stopped explanation as the output is below threshold"
		return(ds)
	end
	dafs, pruning_mask = dafstats(ds, model, n)

	ii = mapreduce(vcat, enumerate(dafs)) do (i,d)
		ii = [(i,j) for j in 1:length(d.daf)]
	end
	# pvalue = mapreduce(d -> Duff.UnequalVarianceTTest(d.daf), vcat, dafs)
	mscore = mapreduce(d -> Duff.meanscore(d.daf), vcat, dafs)

	if method == :uselessfirst
		return(uselessfirst(ds, pruning_mask, model, dafs, mscore, ii, threshold, verbose), pruning_mask)
	else
		@error "unknown pruning method $(method)"
	end
end

"""
	uselessfirst(ds, model, mask, dafs, threshold, verbose)

	Removes items from a sample `ds` such that output of `model(ds)` is above `threshold`.
	Removing starts with items that according to Shapley values do not contribute to the output
"""
function uselessfirst(ds, pruning_mask, model, dafs, score, ii, threshold, verbose)
	mapmask(m -> mask(m) .= true, pruning_mask)

	verbose && println("model output: ", round(minimum(model(ds)), digits = 3))
	for i in sortperm(score, rev = true)
		k,l = ii[i]
		dafs[k][l] = false

		o = minimum(model(prune(ds, pruning_mask)))
		if o <= threshold
			dafs[k][l] = true
		end
		verbose && println((k,l),": score: ",round(score[i], digits = 6),": model output: ",round(o, digits = 3))
	end
	return(prune(ds, pruning_mask))
end
