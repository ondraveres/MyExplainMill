struct ConstExplainer
	α::Float32
end

ConstExplainer() = ConstExplainer(1f0)

function stats(e::ConstExplainer, ds, model, classes, clustering::typeof(_nocluster))
	stats(e, ds, model)
end

function stats(e::ConstExplainer, ds, model)
	create_mask_structure(ds, d -> HeuristicMask(e.α .* ones(Float32, d)))
end

