function explain(ds, f, clustering_model;  n = 1000, pruning = :importantfirst, threshold = 0, clustering = true)
	explain(ds, f, clustering_model, n, pruning, threshold, clustering)
end

function explain(ds, predictor_fun, clustering_model, n, pruning, threshold, clustering)
	if minimum(predictor_fun(ds)) < threshold
		@info "stopped explanation as the output is below threshold"
		return(nothing)
	end
	pruning_mask = @timeit to "dafstats" dafstats(ds, predictor_fun, n, clustering_model, clustering)

	flatmask = FlatView(pruning_mask)
	significance = map(x -> Duff.meanscore(x.mask.daf), flatmask)

	@info "Score estimation failed on $(sum(isnan.(significance))) out of $(length(significance))"

	f = () -> sum(min.(output(predictor_fun(prune(ds, pruning_mask))) .- threshold, 0))
	@info "output - threshold before explanation: $(round(f(), digits = 3))"
	@info "total number of feature: $(length(flatmask))"
	@timeit to "importantfirst!" importantfirst!(f, flatmask, significance)
	@info "output after explanation (should be zero): $(f())"
	pruning_mask
end

function importantfirst!(f, flatmask, significance)
	fill!(flatmask, false)
	previous = f()
	@info "output on empty sample = $previous"
	previous == 0 && return()
	i  = 0
	changed = false
	used = Int[]
	ii = sortperm(significance, rev = false);
	while previous < 0 && i < 10 && !changed
		i += 1
		changed = addminimum!(f, flatmask, ii, strict_improvement = previous < 10)
		used = useditems(flatmask)
		previous = f()
		@info "$(i): output = $(previous) added $(length(used)) features"
	end
	changed = addminimum!(f, flatmask, ii, strict_improvement = false)
	used = useditems(flatmask)
	previous = f()
	@info "$(i): output = $(previous) added $(length(used)) features"
	removeexcess!(f, flatmask, ii[used])
	@info "keeping $(length(used)) features"
end

function removeexcess!(f, flatmask, ii =  1:length(flatmask))
	@debug "enterring removeexcess"
	previous =  f()
	previous < 0 && return(false)
	changed = false
	for i in ii
		flatmask[i] == false && continue
		flatmask[i] = false
		o = f()
		if o < 0
			flatmask[i] = true
		end
		changed = true
		@debug i = i f = o
	end
	return(changed)
end

function addminimum!(f, flatmask, ii = 1:length(flatmask); strict_improvement::Bool = true)
	@debug "enterring addminimum"
	changed = false
	previous =  f()
	previous > 0 && return(changed)
	for i in ii
		all(flatmask[i]) && continue
		flatmask[i] = true
		o = f()
		if strict_improvement && o <= previous
			flatmask[i] = false
		else 
			previous = o
			changed = true
		end
		@debug i = i f = o
		if o >= 0
			break
		end
	end
	return(changed)
end

