struct StochasticExplainer
end

function stats(e::StochasticExplainer, ds, model, classes = onecold(model, ds), clustering = _nocluster)
	statsf(e, ds, model, classes, clustering)
end

function statsf(e::StochasticExplainer, ds, model, f, ::typeof(_nocluster))
	create_mask_structure(ds, d -> HeuristicMask(rand(Float32, d)))
end
