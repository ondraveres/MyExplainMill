struct GradGreedyExplainer

end

function explain(e::GradGreedyExplainer, ds::AbstractNode, model::AbstractMillModel, class::Int; clustering = _nocluster, rel_tol = 0.99f0, partial_evaluation = true, kwargs...)
	gradient_submodular_lbyl(model, ds, class, clustering, rel_tol, partial_evaluation)
end

function gradient_submodular_flat!(f, level_fv)
	level_mk = level_fv.masks
	#this is the core of the grad Gadd
	ps = Flux.Params(map(x -> x.mask.stats, level_mk))
	foreach(p -> fill!(p, 0), ps)
	f₀ = f()
	iter = 0
	# @info "f₀ = $(f₀)"
	while(true)
		gs = gradient(() -> -f(), ps)
		scores = reduce(vcat, map(x -> gs[x.mask.stats][:], level_mk))
		changed = false
		for i in sortperm(scores)
			a = level_fv.itemmap[i]
			level_mk[a.maskid].mask.stats[a.innerid] == 1 && continue
			level_mk[a.maskid].mask.stats[a.innerid] = 1
			fᵢ = f()
			if fᵢ > f₀
				# @info "added $(i) improving to $(fᵢ)"
				f₀ = fᵢ
				changed = true
				break;
			end
			# @info "skipped $(i) $(fᵢ)"
			level_mk[a.maskid].mask.stats[a.innerid] = 0
		end
		iter += 1
		f₀ > 0 && break 
		!changed && break
	end
	for m in level_mk
		m.mask.mask .= m.mask.stats[:] .> 0.5
	end
	f₀
end

function gradient_submodular_lbyl(model, ds, class, clustering, rel_tol, partial_evaluation)
	mk = ExplainMill.Mask(ds, model, d -> fill(1f0, d, 1), clustering)
	parents = parent_structure(mk)
	parents = filter(x -> !isa(x.first, AbstractNoMask), parents)
	if isempty(parents) 
		@warn "Cannot explain empty samples"
		return()
	end
	max_depth = maximum(x.second for x in parents)
	fullmask = FlatView(map(first, parents))
	fill!(fullmask, true)

	hard_f = () -> logsoftmax(model(ds[mk]).data)[class]
	soft_f = () -> logsoftmax(model(ds, mk).data)[class]
	τ = soft_f() + log(rel_tol)
	f₀ = exp(hard_f())
	for j in 1:max_depth
		level_mk = map(first, filter(i -> i.second == j, parents))
		isempty(level_mk) && continue
		level_fv = FlatView(level_mk)

		if partial_evaluation
			sub_model, sub_ds, sub_mk, changed = Mill.partialeval(model, ds, mk, level_mk)
			sub_f = () -> logsoftmax(sub_model(sub_ds, sub_mk).data)[class] - τ
			gradient_submodular_flat!(sub_f, level_fv)
		else
			gradient_submodular_flat!(() -> soft_f() - τ, level_fv)
		end

		fₛ = exp(soft_f())
		fₕ = exp(hard_f())
		@info "level: $(j)  soft f:  = $(fₛ) hard f: $(fₕ) with $(length(useditems(level_fv))) items"

		greedyremoval!(() -> hard_f() - τ, level_fv)
		for m in level_mk
			m.mask.stats .= m.mask.mask
		end
		fₛ = exp(soft_f())
		fₕ = exp(hard_f())
		@info "level: $(j) after rr soft f:  = $(fₛ) hard f: $(fₕ) with $(length(useditems(level_fv))) items"
	end

	used = useditems(fullmask)
	@info "Explanation uses $(length(used)) items out of $(length(fullmask)) at $(exp(hard_f()))"
	mk
end