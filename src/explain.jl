struct DafMask{M}
	daf::Daf 
	mask::M
end

DafMask(m::Mask{Nothing}) = DafMask(Daf(length(m.mask)), m)
DafMask(m::Mask{Vector{Int64}}) = DafMask(Daf(length(unique(m.cluster_membership))), m)

function Duff.update!(d::DafMask{M}, v) where {M<:Mask{Nothing}}
	Duff.update!(d.daf, v, mask(d.mask), participate(d.mask))
end

function Duff.update!(d::DafMask{M}, v) where {M<:Mask{Vector{Int64}}}
	Duff.update!(d.daf, v, mask(d.mask), participate(d.mask), d.mask.cluster_membership)
end

function StatsBase.sample!(m::Mask{Nothing})
	mask(m) .= sample([true, false], length(mask(m)))
end

function StatsBase.sample!(m::Mask{Vector{Int64}})
	ci = m.cluster_membership
	_mask = sample([true, false], maximum(ci))
	for (i,k) in enumerate(ci)
		m.mask[i] = _mask[k]
	end 
end

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
		m != nothing && push!(dafs, DafMask(m))
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
	
	catmask = CatView(tuple([mask(d.mask) for d in dafs]...))
	pvalue = CatView(tuple([Duff.UnequalVarianceTTest(d.daf) for d in dafs]...))
	pvalue = CatView(tuple([Duff.meanscore(d.daf) for d in dafs]...))

	if method == :uselessfirst
		return(uselessfirst(ds, pruning_mask,model, catmask, pvalue, threshold, verbose))
	else
		@error "unknown pruning method $(method)"
	end
end

"""
	uselessfirst(ds, model, mask, dafs, threshold, verbose)

	Removes items from a sample `ds` such that output of `model(ds)` is above `threshold`.
	Removing starts with items that according to Shapley values do not contribute to the output
"""
function uselessfirst(ds, pruning_mask, model, catmask, pvalue, threshold, verbose)
	catmask .= true
	verbose && println("model output: ", round(minimum(model(ds)), digits = 3))
	for i in sortperm(pvalue, rev = true)
		catmask[i] = false
		o = minimum(model(prune(ds, pruning_mask)))
		if o <= threshold
			catmask[i] = true
		end
		verbose && println(i,": p-value: ",round(pvalue[i], digits = 6),": model output: ",round(o, digits = 3))
	end
	return(prune(ds, pruning_mask))
end
