"""
	Fatima, Shaheen S., Michael Wooldridge, and Nicholas R. Jennings. "A linear approximation method for the Shapley value." Artificial Intelligence 172.14 (2008): 1673-1699.

	The current implementation is simplified, as it does not track to which 
	output item of the output of the model an given item in mask belongs to. 
	This has a negative effect on simultaneous explanation of sets of samples. 
"""
struct DafExplainer
	n::Int
	hard::Bool
	banzhaf::Bool
end

DafExplainer(n::Int) = DafExplainer(n, true, false)
DafExplainer() = DafExplainer(200)
ShapExplainer(n::Int) = DafExplainer(n, true, false)
ShapExplainer() = DafExplainer(200)
BanzExplainer(n::Int) = DafExplainer(n, true, true)
BanzExplainer() = BanzExplainer(200)

struct DafMask <: AbstractVectorMask
	x::Vector{Bool}
	stats::Duff.Daf
end

DafMask(d::Int) = DafMask(fill(true, d), Duff.Daf(d))
prunemask(m::DafMask) = m.x
diffmask(m::DafMask) = m.x
simplemask(m::DafMask) = m
Base.length(m::DafMask) = length(m.x)
Base.getindex(m::DafMask, i) = m.x[i]
Base.setindex!(m::DafMask, v, i) = m.x[i] = v
Base.materialize!(m::DafMask, v::Base.Broadcast.Broadcasted) = m.x .= v
heuristic(m::DafMask) = Duff.meanscore(m.stats)

function stats(e::DafExplainer, ds::AbstractNode, model::AbstractMillModel)
	classes = Flux.onecold(softmax(model(ds).data))
	stats(e, ds, model, classes, _nocluster)
end

function stats(e::DafExplainer, ds::AbstractNode, model::AbstractMillModel, classes, cluster::typeof(_nocluster))
	mk = create_mask_structure(ds, d -> ParticipationTracker(DafMask(d)))
	y = gnntarget(model, ds, classes)
	dafstats!(e, mk) do 
		sum(softmax(model(ds[mk]).data) .* y)
	end
	mk
end

function dafstats!(f, e::DafExplainer, mk::AbstractStructureMask)
	for _ in 1:e.n
		sample!(mk)
		updateparticipation!(mk)
		o = f()
		foreach_mask(mk) do m, _
			Duff.update!(e, m, o)
		end
	end
end

function Duff.update!(e::DafExplainer, mk::AbstractVectorMask, o)
	d = simplemask(mk)
	for i in 1:length(d.x)
		!e.banzhaf && !participate(mk)[i] && continue
		Duff.update!(d.stats, o, d.x[i] & participate(mk)[i], i)
	end
end

function StatsBase.sample!(mk::AbstractStructureMask)
	foreach_mask(mk) do m, l
		m .= sample([true,false], length(m))
	end
end