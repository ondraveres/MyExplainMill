output(ds::ArrayNode) = ds.data
output(x::AbstractArray) = x

function StatsBase.sample!(pruning_mask::AbstractExplainMask)
	mapmask(sample!, pruning_mask)
end

function Duff.update!(dafs::Vector, v::Mill.ArrayNode, pruning_mask)
	Duff.update!(dafs, v.data, pruning_mask)
end

function Duff.update!(dafs::Vector, v::AbstractArray{T}, pruning_mask) where{T<:Real}
	for d in dafs 
		Duff.update!(d, v)
	end
end

"""
	updatesamplemembership!(pruning_mask, n)

	this updates the mapping of mask bit to corresponding item in the output,
	which is important for minibatch processing
"""
function updatesamplemembership!(pruning_mask, n)
	for i in 1:n 
		mapmask(pruning_mask) do m 
			participate(m) .= true
		end
		invalidate!(pruning_mask,setdiff(1:n, i))
		mapmask(pruning_mask) do m 
			m.outputid[participate(m)] .= i
		end
	end
end

"""
	function dafstats(ds, model, n=10000)

	Shapley values of individual items of a sample `ds` in the model `model` estimated from `n` trials
"""
function dafstats(ds::AbstractNode, model::AbstractMillModel, i, n, clustering_model, clustering)
	f(x) = output(model(x))[i,:]
	dafstats(ds::AbstractNode, f, n, clustering_model, clustering)
end

function dafstats(ds::AbstractNode, f, n, clustering_model, clustering::Bool)
	clustering = clustering ? (m, ds) -> dbscan_cosine(m(ds).data, 0.1) : _noclustering
	dafstats(ds, f, n, clustering_model, clustering)
end

function dafstats(ds::AbstractNode, f, n, clustering_model, clustering)
	pruning_mask = Mask(ds, clustering_model, Daf, clustering)
	dafstats(pruning_mask, ds, f, n)
end

function dafstats(pruning_mask::AbstractExplainMask, ds, f, n)
	updatesamplemembership!(pruning_mask, nobs(ds))
	dafs = []
	mapmask(pruning_mask) do m
		m != nothing && push!(dafs, m)
	end
	for _j in 1:n
		@timeit to "sample!" sample!(pruning_mask)
		pruned_ds = @timeit to "prune" prune(ds, pruning_mask)
		o = @timeit to "evaluate" f(pruned_ds)
		@timeit to "update!" Duff.update!(dafs, o, pruning_mask)
	end
	return(pruning_mask)
end
