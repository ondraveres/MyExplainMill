using GLMNet
struct LimeExplainer
    schema::Any
    extractor::Any
    perturbation_count::Integer
    perturbation_chance::Real
    perturbation_strategy::String
end

function stats(e::LimeExplainer, ds, model, classes=onecold(model, ds), clustering=_nocluster)
    statsf(e, ds, model, classes, clustering)
end

function statsf(e::LimeExplainer, ds, model, f, ::typeof(_nocluster))
    println("e: ", e)
    println("ds: ", ds)
    println("model: ", model)
    println("f: ", f)
    heuristic_mask = ExplainMill.treelime(ds, model, e.extractor, e.schema, e.perturbation_count, e.perturbation_chance, e.perturbation_strategy)
    fv = FlatView(heuristic_mask)
    fv_v = [fv[i] for i in 1:length(fv.itemmap)]

    println("heuristic_mask ", fv_v, " heures ", heuristic(fv))

    return heuristic_mask
    # create_mask_structure(ds, d -> HeuristicMask(ones(Float32, d)))
end