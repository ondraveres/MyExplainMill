# This will be a PoC of the AbstractMask behavior

struct GradientMask{T<:Number} <: AbstractVectorMask
	x::Vector{T}
end

prunemask(m::GradientMask) = m.x .> 0.5
diffmask(m::GradientMask) = m.x
simplemask(m::GradientMask) = m
Base.length(m::GradientMask) = length(m.x)
Base.getindex(m::GradientMask, i) = m.x[i] .> 0.5
Base.setindex!(m::GradientMask, v, i) = m.x[i] = v

struct GradGreedyExplainer

end

function explain(e::GradGreedyExplainer, ds::AbstractNode, model::AbstractMillModel, class::Int; clustering = _nocluster, rel_tol = 0.99f0, partial_evaluation = true, kwargs...)
	gradient_submodular_lbyl(model, ds, class, clustering, rel_tol, partial_evaluation)
end

function gradient_submodular_flat!(f, fv)
	level_mk = fv.masks
	ps = Flux.Params(map(diffmask ∘ simplemask, level_mk))
	foreach(p -> fill!(p, 0), ps)
	f₀ = f()
	iter = 0
	while(true)
		gs = gradient(() -> -f(), ps)
		scores = reduce(vcat, map(m -> gs[simplemask(m).x][:], level_mk))
		changed = false
		for i in sortperm(scores)
			fv[i] == 1 && continue
			fv[i] = 1
			fᵢ = f()
			if fᵢ > f₀
				f₀ = fᵢ
				changed = true
				break;
			end
			fv[i] = 0
		end
		iter += 1
		f₀ > 0 && break 
		!changed && break
	end
	f₀
end

function gradient_submodular_flat(model, ds, class, clustering, rel_tol, partial_evaluation)
	create_mask = d -> ParticipationTracker(GradientMask(ones(Float32, d)))
	mk = create_mask_structure(ds, create_mask)
	full_flat = FlatView(mk)
	full_flat .= true
	y = gnntarget(model, ds, class)
	
	hard_f = () -> Flux.Losses.logitcrossentropy(model(ds[mk]).data, y)
	soft_f = () -> Flux.Losses.logitcrossentropy(model(ds, mk).data, y)
	τ = soft_f() + log(rel_tol)
	f₀ = exp(hard_f())
	gradient_submodular_flat!(() -> soft_f() - τ, full_flat)

	fₛ = exp(soft_f())
	fₕ = exp(hard_f())
	@info "soft f:  = $(fₛ) hard f: $(fₕ) with $(length(useditems(full_flat))) items"


	greedyremoval!(() -> hard_f() - τ, full_flat)
	fₛ = exp(soft_f())
	fₕ = exp(hard_f())
	@info "after rr soft f:  = $(fₛ) hard f: $(fₕ) with $(length(useditems(full_flat))) items"

	used = useditems(full_flat)
	@info "Explanation uses $(length(used)) items out of $(length(full_flat)) at $(exp(hard_f()))"
	mk
end

function gradient_submodular_lbyl(model, ds, class, clustering, rel_tol, partial_evaluation)
	create_mask = d -> ParticipationTracker(GradientMask(ones(Float32, d)))
	mk = create_mask_structure(ds, create_mask)
	all_masks = collect_masks_with_levels(mk)
	if isempty(all_masks) 
		@warn "Cannot explain empty samples"
		return()
	end
	
	full_flat = FlatView(map(first, all_masks))
	full_flat .= true
	y = gnntarget(model, ds, class)
	

	hard_f = () -> Flux.Losses.logitcrossentropy(model(ds[mk]).data, y)
	soft_f = () -> Flux.Losses.logitcrossentropy(model(ds, mk).data, y)
	τ = soft_f() + log(rel_tol)
	f₀ = exp(hard_f())
	for level in 1:maximum(map(x -> x.second, all_masks))
		updateparticipation!(mk)
		full_flat .= copy2vec(full_flat) .& participate(full_flat)
		level_masks = map(first, filter(m -> m.second == level, all_masks))
		level_fv = FlatView(level_masks)
		isempty(level_masks) && continue

		if partial_evaluation
			modelₗ, dsₗ, mkₗ = partialeval(model, ds, mk, level_masks)
			sub_f = () -> Flux.Losses.logitcrossentropy(modelₗ(dsₗ, mkₗ).data,  y) - τ
			gradient_submodular_flat!(sub_f, level_fv)
		else
			gradient_submodular_flat!(() -> soft_f() - τ, level_fv)
		end

		fₛ = exp(soft_f())
		fₕ = exp(hard_f())
		@info "level: $(level)  soft f:  = $(fₛ) hard f: $(fₕ) with $(length(useditems(level_fv))) items"

		greedyremoval!(() -> hard_f() - τ, level_fv)
		fₛ = exp(soft_f())
		fₕ = exp(hard_f())
		@info "level: $(level) after rr soft f:  = $(fₛ) hard f: $(fₕ) with $(length(useditems(level_fv))) items"
	end

	used = useditems(full_flat)
	@info "Explanation uses $(length(used)) items out of $(length(full_flat)) at $(exp(hard_f()))"
	mk
end