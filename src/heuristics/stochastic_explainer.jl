struct StochasticExplainer
end

function stats(e::StochasticExplainer, ds, model, i, n, clustering = ExplainMill._nocluster)
	create_mask_structure(ds, d -> HeuristicMask(rand(Float32, d)))
end

function stats(e::StochasticExplainer, ds, model)
	create_mask_structure(ds, d -> HeuristicMask(rand(Float32, d)))
end

stats(e::StochasticExplainer, ds, model, classes) = stats(e, ds, model)
