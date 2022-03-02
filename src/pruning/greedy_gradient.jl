# This will be a PoC of the AbstractMask behavior
struct GreedyGradient
end

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

function explain(e::GradGreedyExplainer, ds::AbstractMillNode, model::AbstractMillModel, class::Int; clustering = _nocluster, rel_tol = 0.99f0, partial_evaluation = true, kwargs...)
	greedy_gradient_lbyl(model, ds, class, clustering, rel_tol, partial_evaluation)
end

function greedy_gradient_flat!(f, fv)
	level_mk = fv.masks
	ps = Flux.Params(map(diffmask ∘ simplemask, level_mk))
	fv .= 0
	f₀ = f()
	@info "starting greedy gradient ascend with $(f₀)"
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

function greedy_gradient_lbyl!(model, ds, mk, class, thresholds, partial_evaluation, random_removal)
	all_masks = collect_masks_with_levels(mk)
	full_flat = FlatView(map(first, all_masks))
	full_flat .= true
	y = gnntarget(model, ds, class)
	
	hard_f = () -> sum(min.(logitconfgap(model, ds[mk], class) .- thresholds, 0))
	soft_f = () -> sum(min.(logitconfgap(model(ds, mk).data, class) .- thresholds, 0))
	f₀ = hard_f()
	for level in 1:maximum(map(x -> x.second, all_masks))
		updateparticipation!(mk)
		full_flat .= copy2vec(full_flat) .& participate(full_flat)
		level_masks = map(first, filter(m -> m.second == level, all_masks))
		level_fv = FlatView(level_masks)
		isempty(level_masks) && continue

		if partial_evaluation
			modelₗ, dsₗ, mkₗ = partialeval(model, ds, mk, level_masks)
			sub_f = () -> sum(min.(logitconfgap(modelₗ(dsₗ, mkₗ).data, class) .- thresholds, 0))
			greedy_gradient_flat!(sub_f, level_fv)
		else
			greedy_gradient_flat!(soft_f, level_fv)
		end

		@assert model(ds, mk).data ≈ model(ds[mk]).data "multiplicative subset and hard subset disagrees"
		fₛ = soft_f()
		fₕ = hard_f()
		@info "level: $(level)  soft f:  = $(fₛ) hard f: $(fₕ) with $(length(useditems(level_fv))) items"

		if random_removal 
			greedyremoval!(hard_f, level_fv)
			fₛ = soft_f()
			fₕ = hard_f()
			@info "level: $(level) after rr soft f:  = $(fₛ) hard f: $(fₕ) with $(length(useditems(level_fv))) items"
		end
	end

	used = useditems(full_flat)
	@info "Explanation uses $(length(used)) items out of $(length(full_flat)) at $(exp(hard_f()))"
	mk
end

function greedy_gradient_lbyl(ds; clustering=_nocluster, rel_tol=nothing, abs_tol=nothing, partial_evaluation = true, adjust_mask=identity)
	class = Flux.onecold(softmax(model(ds).data))
	greedy_gradient_lbyl(model, ds, class, clustering, abs_tol, rel_tol, partial_evaluation, adjust_mask)
end

function explain(e::GreedyGradient, ds::AbstractMillNode, model::AbstractMillModel, class; clustering = ExplainMill._nocluster, pruning_method=:LbyL_HArr,
        abs_tol=nothing, rel_tol=nothing, adjust_mask = identity)
    cg = logitconfgap(model, ds, class)
    @assert all(0 .≤ cg) "Cannot explain class with negative confidence gap!"
	create_mask = d -> ParticipationTracker(GradientMask(ones(Float32, d)))
	mk = create_mask_structure(ds, create_mask)
    mk = adjust_mask(mk)
	all_masks = collect_masks_with_levels(mk)
	if isempty(all_masks) 
		@warn "Cannot explain empty samples"
		return()
	end	

    thresholds = get_thresholds(cg, abs_tol, rel_tol)
    if pruning_method ∈ (:LbyL_HArr, :LbyL_HAdd, :LbyLo_HArr, :LbyLo_HAdd)
    	random_removal = pruning_method ∈ (:LbyL_HArr, :LbyLo_HArr)
    	partial_evaluation = pruning_method ∈ (:LbyLo_HAdd, :LbyLo_HArr)
	    greedy_gradient_lbyl!(model, ds, mk, class, thresholds, partial_evaluation, random_removal)
	else
		error("GreedyGradient supports only [:LbyL_HArr, :LbyL_HAdd] pruning functions. Flat methods are not supported at all, because the")
	end

    mk
end