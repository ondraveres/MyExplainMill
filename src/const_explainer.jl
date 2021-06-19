using Zygote, Setfield

struct ConstExplainer
	α::Float32
end

ConstExplainer() = ConstExplainer(1f0)

function stats(e::ConstExplainer, ds, model, i, n, clustering = ExplainMill._nocluster)
	ExplainMill.Mask(ds, model, d -> e.α .* ones(Float32, d), clustering)
end

scorefun(e::ConstExplainer, x::AbstractStructureMask) = x.mask.stats
