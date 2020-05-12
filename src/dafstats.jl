struct DafExplainer
	hard::Bool
end

DafExplainer() = DafExplainer(true)

function stats(e::DafExplainer, ds::AbstractNode, model::AbstractMillModel, i::Int, n, clustering = ExplainMill._nocluster; threshold = 0.1)
	soft_model = (ds...) -> softmax(model(ds...));
	f = e.hard ? (ds, ms) -> output(soft_model(prune(ds, ms)))[i,:] : (ds, ms) -> output(soft_model(ds, ms))[i,:]
	stats(e, ds, model, f, n, clustering)
end

function stats(e::DafExplainer, ds::AbstractNode, model::AbstractMillModel, f, n, clustering = ExplainMill._nocluster; threshold = 0.1)
	ms = ExplainMill.Mask(ds, model, Duff.Daf, clustering)
	updatesamplemembership!(ms, nobs(ds))
	@timeit to "dafstats" dafstats(ms, () -> f(ds, ms), n)
end

function dafstats(pruning_mask::AbstractExplainMask, f, n)
	dafs = []
	mapmask(pruning_mask) do m
		m != nothing && push!(dafs, m)
	end
	for _j in 1:n
		@timeit to "sample!" sample!(pruning_mask)
		updateparticipation!(pruning_mask)
		o = @timeit to "evaluate" f()
		@timeit to "update!" Duff.update!(dafs, o, pruning_mask)
	end
	return(pruning_mask)
end

scorefun(e::DafExplainer, x::AbstractExplainMask) = Duff.meanscore(x.mask.stats)
scorefun(e::DafExplainer, x::Mask) = Duff.meanscore(x.stats)

function StatsBase.sample!(pruning_mask::AbstractExplainMask)
	mapmask(sample!, pruning_mask)
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
