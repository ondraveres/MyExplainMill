"""
Fatima, Shaheen S., Michael Wooldridge, and Nicholas R. Jennings. "A linear approximation method for the Shapley value." Artificial Intelligence 172.14 (2008): 1673-1699.
"""
struct DafExplainer
	n::Int
	hard::Bool
	banzhaf::Bool
end

DafExplainer(n::Int) = DafExplainer(n, true, false)
DafExplainer() = DafExplainer(200)
BanzExplainer(n::Int) = DafExplainer(n, true, true)
BanzExplainer() = DafExplainer(200)

function stats(e::DafExplainer, ds::AbstractNode, model::AbstractMillModel, i::Int, clustering = ExplainMill._nocluster; threshold = 0.1)
	soft_model = (ds...) -> softmax(model(ds...));
	f = e.hard ? (ds, ms) -> output(soft_model(ds[ms]))[i,:] : (ds, ms) -> output(soft_model(ds, ms))[i,:]
	stats(e, ds, model, f, clustering)
end

function stats(e::DafExplainer, ds::AbstractNode, model::AbstractMillModel, f, clustering = ExplainMill._nocluster; threshold = 0.1)
	ms = ExplainMill.Mask(ds, model, Duff.Daf, clustering)
	updatesamplemembership!(ms, nobs(ds))
	@timeit to "dafstats" dafstats(e, ms, () -> f(ds, ms))
end

function dafstats(e::DafExplainer, pruning_mask::AbstractExplainMask, f)
	dafs = []
	mapmask(pruning_mask) do m
		m != nothing && push!(dafs, m)
	end
	for _j in 1:e.n
		@timeit to "sample!" sample!(pruning_mask)
		updateparticipation!(pruning_mask)
		o = @timeit to "evaluate" f()
		@timeit to "update!" Duff.update!(e, dafs, o, pruning_mask)
	end
	return(pruning_mask)
end

function Duff.update!(e::DafExplainer, dafs::Vector, v::AbstractArray{T}, pruning_mask) where{T<:Real}
	for d in dafs 
		Duff.update!(e, d, v)
	end
end

function Duff.update!(e::DafExplainer, d::Mask, v::AbstractArray)
	s = d.stats
	for i in 1:length(d.mask)
		!e.banzhaf && !d.participate[i] && continue
		f = v[d.outputid[i]]
		j = _cluster_membership(d.cluster_membership, i)
		Duff.update!(s, f, d.mask[i] & d.participate[i], j)
	end
end

scorefun(e::DafExplainer, x::AbstractExplainMask) = Duff.meanscore(x.mask.stats)
scorefun(e::DafExplainer, x::Mask) = Duff.meanscore(x.stats)

function StatsBase.sample!(pruning_mask::AbstractExplainMask)
	mapmask(sample!, pruning_mask)
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
