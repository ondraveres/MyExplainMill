struct ConstExplainer
	α::Float32
end

ConstExplainer() = ConstExplainer(1f0)

function stats(e::ConstExplainer, ds, model, classes = onecold(model, ds), clustering = _nocluster)
	statsf(e, ds, model, classes, clustering)
end

function statsf(e::ConstExplainer, ds, model, f, ::typeof(_nocluster))
	create_mask_structure(ds, d -> HeuristicMask(e.α .* ones(Float32, d)))
end
