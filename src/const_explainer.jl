using Zygote, HierarchicalUtils, Setfield

struct ConstExplainer
	α::Float32
end

ConstExplainer() = ConstExplainer(1f0)

function stats(e::ConstExplainer, ds, model, i, n, clustering = ExplainMill._nocluster; threshold = 0.1)
	ExplainMill.Mask(ds, model, d -> e.α .* ones(Float32, d), clustering)
end

scorefun(e::ConstExplainer, x::AbstractExplainMask) = x.mask.stats