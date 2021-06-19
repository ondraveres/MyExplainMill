using Zygote, Setfield

struct StochasticExplainer
end

function stats(e::StochasticExplainer, ds, model, i, n, clustering = ExplainMill._nocluster)
	ExplainMill.Mask(ds, model, d -> rand(Float32, d), clustering)
end

scorefun(e::StochasticExplainer, x::AbstractStructureMask) = x.mask.stats
