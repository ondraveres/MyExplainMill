using Zygote, Setfield

struct ConstExplainer
	α::Float32
end

ConstExplainer() = ConstExplainer(1f0)

function stats(e::ConstExplainer, ds, model, i, n, clustering = ExplainMill._nocluster)
	create_mask_structure(ds, d -> SimpleMask(e.α .* ones(Float32, d)))
end

scorefun(e::ConstExplainer, x::AbstractStructureMask) = x.mask.x
