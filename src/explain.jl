function get_thresholds(cg, abs_tol, rel_tol)
    if isnothing(abs_tol) && isnothing(rel_tol)
        @warn "No tolerance specified, setting rel_tol=0.9"
        rel_tol = 0.9
    end
    if isnothing(abs_tol)
        @assert 0 ≤ rel_tol ≤ 1 "Relative tolerance must be in [0, 1]!"
        rel_tol .* cg
    else
        @assert all(abs_tol .≤ cg) "Absolute tolerance must be smaller than the confidence gap!"
        cg .- abs_tol
    end
end

"""
	explain(e, ds::AbstractNode, model::AbstractMillModel, i, n, clustering = ExplainMill._nocluster; threshold = nothing, pruning_method=:LbyL_HArr, gap = 0.9f0)

	explain mask of a sample(s) ds, such that the confidencegap of the explanation is either above `threshold` (if set)
	or above the `gap*confidencegap` of the full sample(s) with `gap` being the `0.9` by default.
	i is the index of the class which we are explaining and `n` is the number of repetitions / gradient
	iterations in the calculation of stats.
"""
function explain(e, ds::AbstractNode, model::AbstractMillModel, class; clustering = ExplainMill._nocluster, pruning_method=:LbyL_HArr,
        abs_tol=nothing, rel_tol=nothing, adjust_mask = identity)
    cg = logitconfgap(model, ds, class)
    @assert all(0 .≤ cg) "Cannot explain class with negative confidence gap!"
    mk = stats(e, ds, model, class, clustering)
    mk = adjust_mask(mk)
    thresholds = get_thresholds(cg, abs_tol, rel_tol)
    prune!(mk, model, ds, class, thresholds, pruning_method)
    mk
end

function explain(e, ds::AbstractNode, model::AbstractMillModel; kwargs...)
    class = Flux.onecold(softmax(model(ds).data))
    if length(unique(class))  > 1 
    	@warn "Two or more classes predicted by the model!, wish you know what you are doing."
    end
    explain(e, ds, model, class; kwargs...)
end
