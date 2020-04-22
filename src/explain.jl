Base.minimum(ds::ArrayNode) = minimum(ds.data)

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
	explain(ds, model, i;  n = 1000, pruning = :importantfirst, scoring = :mean, threshold = 0.5, verbose = false, clustering = true, completely = false, clustering_model = model)

	Explain sample `ds` by removing its items such that output of `model(ds)` is above `threshold`.
	`method` controls the order in which the items are subjected to iterative removal. 
	:importantlast means that samples with low importance are removed first

"""
function explain(ds, model, i;  n = 1000, pruning = :importantfirst, scoring = :mean, threshold = 0.5, verbose = false, clustering = true, completely = false, clustering_model = model)
	completely ? explaincompletely(ds, model, clustering_model, i, n, pruning, getscorefun(scoring), threshold, verbose, clustering) : explain(ds, model, clustering_model, i, n, pruning, getscorefun(scoring), threshold, verbose, clustering)
end

function explain(ds, model, clustering_model,  i, n, pruning, scorefun, threshold, verbose, clustering)
	if minimum(output(model(ds))[i,:]) < threshold
		@info "stopped explanation as the output is below threshold"
		return(nothing)
	end
	dafs, pruning_mask = @timeit to "dafstats" dafstats(ds, model, i, n, clustering_model, clustering, verbose)

	ii = mapreduce(vcat, enumerate(dafs)) do (i,d)
		[(i,j) for j in 1:length(d.daf)]
	end
	mscore = mapreduce(d -> scorefun(d.daf), vcat, dafs)
	verbose && println("Score estimation failed on ", sum(isnan.(mscore))," out of ",length(mscore))

	f = x -> sum(min.(output(model(x))[i,:] .- threshold, 0))
	verbose && println("model output before explanation: ", round(f(ds), digits = 3))
	verbose && println("total number of feature: ", length(ii))
	if pruning == :importantlast
			@timeit to "importantlast" importantlast(f, ds, pruning_mask, dafs, mscore, ii, verbose)
		elseif pruning == :importantfirst
			@timeit to "importantfirst"  importantfirst(f, ds, pruning_mask, dafs, mscore, ii, verbose)
		else
			@error "unknown pruning $(pruning)"
	end
	# ex_ds = prune(ds, pruning_mask)
	# ex_ds = ex_ds[1:nobs(ex_ds)]
	# verbose && println("model output after explanation: ", round(f(ex_ds), digits = 3))
	# ex_ds
	pruning_mask
end

function explain2(ds, f, clustering_model;  n = 1000, pruning = :importantfirst, scoring = :mean, threshold = 0, verbose = false, clustering = true)
	explain2(ds, f, clustering_model, n, pruning, getscorefun(scoring), threshold, verbose, clustering)
end
function explain2(ds, predictor_fun, clustering_model, n, pruning, scorefun, threshold, verbose, clustering)
	if minimum(predictor_fun(ds)) < threshold
		@info "stopped explanation as the output is below threshold"
		return(nothing)
	end
	dafs, pruning_mask = @timeit to "dafstats" dafstats(ds, predictor_fun, n, clustering_model, clustering, verbose)

	ii = mapreduce(vcat, enumerate(dafs)) do (i,d)
		ii = [(i,j) for j in 1:length(d.daf)]
	end
	mscore = mapreduce(d -> scorefun(d.daf), vcat, dafs)
	verbose && println("Score estimation failed on ", sum(isnan.(mscore))," out of ",length(mscore))

	f = x -> sum(min.(output(predictor_fun(x)) .- threshold, 0))
	verbose && println("model output before explanation: ", round(f(ds), digits = 3))
	verbose && println("total number of feature: ", length(ii))
	if pruning == :importantlast
			@timeit to "importantlast" importantlast(f, ds, pruning_mask, dafs, mscore, ii, verbose)
		elseif pruning == :importantfirst
			@timeit to "importantfirst"  importantfirst(f, ds, pruning_mask, dafs, mscore, ii, verbose)
		else
			@error "unknown pruning $(pruning)"
	end
	# ex_ds = prune(ds, pruning_mask)
	# ex_ds = ex_ds[1:nobs(ex_ds)]
	# verbose && println("model output after explanation: ", round(f(ex_ds), digits = 3))
	# ex_ds
	pruning_mask
end

function explaincompletely(ds, model, clustering_model, i, n, pruning, scorefun, threshold, verbose, clustering)
	if minimum(output(model(ds))[i,:]) < threshold
		@info "stopped explanation as the output is below threshold"
		return(nothing)
	end
	dafs, pruning_mask = dafstats(ds, model, i, n, clustering)

	ii = mapreduce(vcat, enumerate(dafs)) do (i,d)
		ii = [(i,j) for j in 1:length(d.daf)]
	end
	mscore = mapreduce(d -> scorefun(d.daf), vcat, dafs)

	f = x -> sum(min.(output(model(x))[i,:] .- threshold, 0))
	used = Vector{Int}()
	ex_dss = []
	verbose && println("model output before explanation: ", round(f(ds), digits = 3))
	verbose && println("total number of feature: ", length(ii))
	while length(used) < length(ii)
		if pruning == :importantlast
			@error "not implemented yet, as Pevnak is afraid this to be terribly slow"
		elseif pruning == :importantfirst
			free = setdiff(1:length(ii), used)
			importantfirst(f, ds, pruning_mask, dafs, mscore[free], ii[free], verbose)
			used = findall(map(i -> all(dafs[i[1]][i[2]]), ii))
			ex_ds = prune(ds, pruning_mask)
			verbose && println("model output after explanation: ", round(f(ex_ds), digits = 3))
			round(f(ex_ds), digits = 3) < 0 && break
			push!(ex_dss, ex_ds[1:nobs(ex_ds)])
		else
			@error "unknown pruning $(pruning)"
		end
	end
	ex_dss
end

"""
	importantlast(ds, model, mask, dafs)

	Removes items from a sample `ds` such that output of `model(ds)` is above zero.
	Removing starts with items that according to Shapley values do not contribute to the output
"""
function importantlast(f, ds, pruning_mask, dafs, mscore, ii, verbose::Bool = false)
	mapmask(m -> mask(m) .= true, pruning_mask)
	removeexcess!(f, ds, pruning_mask, dafs, ii[sortperm(mscore, rev = true)])
	prune(ds, pruning_mask)
end

useditems(dafs, ii) = findall(map(i -> all(dafs[i[1]][i[2]]), ii))

function importantfirst(f, ds, pruning_mask, dafs, mscore, ii, verbose::Bool = false)
	mapmask(m -> mask(m) .= false, pruning_mask)
	previous = f(prune(ds, pruning_mask))
	println("output on empty sample = ", previous)
	i  = 0
	changed = false
	used = Int[]
	while previous < 0 && i < 10 && !changed
		i += 1
		changed = addminimum!(f, ds, pruning_mask, dafs, ii[sortperm(mscore, rev = false)], strict_improvement = true)
		used = useditems(dafs, ii)
		previous = f(prune(ds, pruning_mask))
		verbose && println("$(i): output = $(previous) added $(length(used)) features")
	end
	if previous < 0 
		addminimum!(f, ds, pruning_mask, dafs, ii[sortperm(mscore, rev = false)], strict_improvement = false)
		used = useditems(dafs, ii)
		verbose && println("added $(length(used)) features")
	end
	removeexcess!(f, ds, pruning_mask, dafs, ii[used])
	used = findall(map(i -> all(dafs[i[1]][i[2]]), ii))
	verbose && println("keeping $(length(used)) features")
	prune(ds, pruning_mask)
end


function removeexcess!(f, ds, pruning_mask, dafs, ii)
	previous =  f(prune(ds, pruning_mask))
	previous < 0 && return(false)
	changed = false
	for (k,l) in ii
		dafs[k][l] == false && continue
		dafs[k][l] = false
		o = f(prune(ds, pruning_mask))
		if o < 0
			dafs[k][l] = true
			break
		else
			changed = true
		end
	end
	return(changed)
end

function addminimum!(f, ds, pruning_mask, dafs, ii; strict_improvement::Bool = true)
	changed = false
	previous =  f(prune(ds, pruning_mask))
	previous > 0 && return(changed)
	for (k,l) in ii
		all(dafs[k][l]) && continue
		dafs[k][l] = true
		o = f(prune(ds, pruning_mask))
		if strict_improvement && o <= previous
			dafs[k][l] = false
		else 
			previous = o
			changed = true
		end
		if o >= 0
			break
		end
	end
	return(changed)
end
