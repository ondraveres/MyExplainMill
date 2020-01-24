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

function maskstats(pruning_mask)
	k,l = 0,0
	mapmask(pruning_mask) do m
		if m != nothing
			k += length(m.daf)
			l+= length(m.mask)
		end
	end
	println("degrees of freedom: ", k, " number of items: ", l)
end


"""
	function dafstats(ds, model, n=10000)

	Shapley values of individual items of a sample `ds` in the model `model` estimated from `n` trials
"""
function dafstats(ds, model, i, n, clustering)
	pruning_mask = clustering ? Mask(ds, model) : Mask(ds)
	dafs = []
	mapmask(pruning_mask) do m
		m != nothing && push!(dafs, m)
	end
	for _j in 1:n
		@timeit to "sample!" sample!(pruning_mask)
		pruned_ds = @timeit to "prune" prune(ds, pruning_mask)
		o = @timeit to "evaluate" model(pruned_ds)
		@timeit to "update!" Duff.update!(dafs, o.data[i,:], pruning_mask)
	end
	return(dafs, pruning_mask)
end

function getscorefun(s)
	if s == :mean 
		return(Duff.meanscore)
	elseif s == :pvalue
		return(d -> 1 - Duff.pvalue(d))
	else 
		@error "unknown score function $(s)"
	end
end

"""
	explain(ds, model;  n = 10000, method = :importantlast, threshold = 0.5, verbose = false)

	Explain sample `ds` by removing its items such that output of `model(ds)` is above `threshold`.
	`method` controls the order in which the items are subjected to iterative removal. 
	:importantlast means that samples with low importance are removed first
"""
function explain(ds, model, i;  n = 1000, pruning = :importantfirst, scoring = :mean, threshold = 0.5, verbose = false, clustering = true)
	explain(ds, model, i, n, pruning, getscorefun(scoring), threshold, verbose, clustering)
end

function explain(ds, model, i, n, pruning, scorefun, threshold, verbose, clustering)
	if minimum(model(ds).data[i,:]) < threshold
		@info "stopped explanation as the output is below threshold"
		return(ds)
	end
	dafs, pruning_mask = dafstats(ds, model, i, n, clustering)

	ii = mapreduce(vcat, enumerate(dafs)) do (i,d)
		ii = [(i,j) for j in 1:length(d.daf)]
	end

	mscore = mapreduce(d -> scorefun(d.daf), vcat, dafs)

	f = x -> minimum(model(x).data[i,:])
	verbose && println("model output before explanation: ", round(f(ds), digits = 3))
	if pruning == :importantlast
			importantlast(ds, model, i, pruning_mask, dafs, mscore, ii, threshold, verbose)
		elseif pruning == :importantfirst
			importantfirst(ds, model, i, pruning_mask, dafs, mscore, ii, threshold, verbose)
		else
			@error "unknown pruning $(pruning)"
	end
	ex_ds = prune(ds, pruning_mask)
	ex_ds = ex_ds[1:nobs(ex_ds)]
	verbose && println("model output after explanation: ", round(f(ex_ds), digits = 3))
	ex_ds
end

"""
	importantlast(ds, model, mask, dafs, threshold, verbose)

	Removes items from a sample `ds` such that output of `model(ds)` is above `threshold`.
	Removing starts with items that according to Shapley values do not contribute to the output
"""
function importantlast(ds, model, i, pruning_mask, dafs, mscore, ii, threshold, verbose)
	mapmask(m -> mask(m) .= true, pruning_mask)
	f = x -> minimum(model(x).data[i,:])
	removeexcess!(pruning_mask, dafs, ds, f, ii[sortperm(mscore, rev = true)], threshold)
	prune(ds, pruning_mask)
end

function importantfirst(ds, model, i, pruning_mask, dafs, mscore, ii, threshold, verbose)
	mapmask(m -> mask(m) .= false, pruning_mask)
	f = x -> minimum(model(x).data[i,:])
	addminimum!(pruning_mask, dafs, ds, f, ii[sortperm(mscore, rev = true)], threshold)
	removeexcess!(pruning_mask, dafs, ds, f, ii[sortperm(mscore, rev = true)], threshold)
	prune(ds, pruning_mask)
end

function removeexcess!(pruning_mask, dafs, ds, f, ii, threshold)
	minimum(f(prune(ds, pruning_mask))) < threshold && return(threshold)
	changed = false
	for (k,l) in ii
		dafs[k][l] == false && continue
		dafs[k][l] = false
		changed = true
		o = f(prune(ds, pruning_mask))
		if o < threshold
			dafs[k][l] = true
		end
	end
	return(changed)
end

function addminimum!(pruning_mask, dafs, ds, f, ii, threshold)
	minimum(f(prune(ds, pruning_mask))) > threshold && return(threshold)
	changed = false
	for (k,l) in ii
		dafs[k][l] == true && continue
		dafs[k][l] = true
		changed = true
		o = minimum(f(prune(ds, pruning_mask)))
		if o > threshold
			return(changed)
		end
	end
	return(changed)
end
