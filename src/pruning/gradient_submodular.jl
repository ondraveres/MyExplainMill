# This will be a PoC of the AbstractMask behavior

struct GradientMask{T<:Number}
	x::Vector{T}
end

prunemask(m::GradientMask) = m.x .> 0.5
diffmask(m::GradientMask) = m.x
Base.length(m::GradientMask) = length(m.x)
Base.getindex(m::GradientMask, i) = m.x[i] .> 0.5
Base.setindex!(m::GradientMask, v, i) = m.x[i] = v

struct GradGreedyExplainer

end

function explain(e::GradGreedyExplainer, ds::AbstractNode, model::AbstractMillModel, class::Int; clustering = _nocluster, rel_tol = 0.99f0, partial_evaluation = true, kwargs...)
	gradient_submodular_lbyl(model, ds, class, clustering, rel_tol, partial_evaluation)
end

function gradient_submodular_flat!(f, level_fv)
	level_mk = level_fv.masks
	ps = Flux.Params(map(x -> x.mask.x, level_mk))
	foreach(p -> fill!(p, 0), ps)
	f₀ = f()
	iter = 0
	while(true)
		gs = gradient(() -> -f(), ps)
		scores = reduce(vcat, map(x -> gs[x.mask.x][:], level_mk))
		changed = false
		for i in sortperm(scores)
			level_fv[i] == 1 && continue
			level_fv[i] = 1
			fᵢ = f()
			if fᵢ > f₀
				f₀ = fᵢ
				changed = true
				break;
			end
			level_fv[i] = 0
		end
		iter += 1
		f₀ > 0 && break 
		!changed && break
	end
	f₀
end

function gradient_submodular_lbyl(model, ds, class, clustering, rel_tol, partial_evaluation)
	create_mask = d -> GradientMask(ones(Float32, d))
	mk = create_mask_structure(ds, model, create_mask, clustering)
	parents = parent_structure(mk)
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
		fₛ = exp(soft_f())
		fₕ = exp(hard_f())
		@info "level: $(j) after rr soft f:  = $(fₛ) hard f: $(fₕ) with $(length(useditems(level_fv))) items"
	end

	used = useditems(fullmask)
	@info "Explanation uses $(length(used)) items out of $(length(fullmask)) at $(exp(hard_f()))"
	mk
end