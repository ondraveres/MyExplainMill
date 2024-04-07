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
    extractor::Any
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

function stats(e::DafExplainer, ds::AbstractMillNode, model::AbstractMillModel, classes=onecold(model, ds), cluster=_nocluster)
    y = gnntarget(model, ds, classes)
    println("y: ", y)
    f(o) = sum(softmax(o) .* y)
    heuristic_mask = statsf(e, ds, model, f, cluster)
    fv = FlatView(heuristic_mask)
    fv_v = [fv[i] for i in 1:length(fv.itemmap)]

    println("heuristic_mask ", fv_v, " heures ", heuristic(fv))
    return heuristic_mask
end

function statsf(e::DafExplainer, ds::AbstractMillNode, model::AbstractMillModel, f, cluster::typeof(_nocluster))
    mk = create_mask_structure(ds, d -> ParticipationTracker(DafMask(d)))
    dafstats!(e, mk, ds, model) do
        f(model(ds[mk]))
    end
    mk
end

function dafstats!(f, e::DafExplainer, mk::AbstractStructureMask, ds, model)
    flat_modification_masks = []
    labels = []
    distances = []
    for _ in 1:e.n
        random_number = rand()

        sample!(mk, Weights([random_number, 1 - random_number]))

        updateparticipation!(mk)

        flat_view = ExplainMill.FlatView(mk)
        p_flat_view = participate(ExplainMill.FlatView(mk))
        # for i in 1:length(p_flat_view)
        #     if flat_view[i] && p_flat_view[i]
        #         println("MATCH")
        #     else
        #         println("Not match")
        #     end
        # end
        new_mask_bool_vector = [(p_flat_view[i] && flat_view[i]) for i in 1:length(p_flat_view)]

        push!(flat_modification_masks, new_mask_bool_vector)
        push!(labels, argmax(model(ds[mk]))[1])
        println(argmax(model(ds[mk]))[1])
        og_mk = create_mask_structure(ds, d -> SimpleMask(d))

        s = ExplainMill.e2boolean(ds, mk, e.extractor)
        og = ExplainMill.e2boolean(ds, og_mk, e.extractor)

        println(nnodes(s))
        println(nleaves(s))
        ce = jsondiff(og, s)
        ec = jsondiff(s, og)
        println("metric ", nleaves(ce) + nleaves(ec))
        push!(distances, nleaves(ce) + nleaves(ec))

        # o = f()
        # foreach_mask(mk) do m, _
        #     Duff.update!(e, m, o)
        # end
    end
    println(length(flat_modification_masks[1]), flat_modification_masks[1])
    println("labels", labels)

    og_class = Flux.onecold((model(ds)))[1]
    labels = ifelse.(labels .== og_class, 2, 1)
    X = hcat(flat_modification_masks...)
    y = labels

    # println("y is ", y)

    Xmatrix = convert(Matrix{Float64}, X')  # transpose X because glmnet assumes features are in columns
    yvector = convert(Vector{Float64}, y)

    # Fit the model
    label_freq = countmap(yvector)


    # weights = 1 ./ (2 .^ (sum(Xmatrix .== 1, dims=2)[:, 1] ./ 100))
    # weights = sum(Xmatrix .== 1, dims=2)[:, 1]

    weights = [1 / label_freq[label] for label in yvector]
    normalized_distances = 1 ./ ((distances .+ 1e-6) .^ 2)
    # weights /= sum(weights)
    println("weights are", weights .* normalized_distances)

    println(typeof(Xmatrix))
    println(typeof(yvector))

    # Fit glmnet model with weights
    cv = glmnetcv(Xmatrix, yvector; weights=weights .* normalized_distances, alpha=0.0)
    # println("cv", cv.meanloss)

    # Perform cross-validation
    # cv = glmnetcv(fit)

    # βs = cv.path.betas
    # λs = cv.lambda
    # βs

    # sharedOpts = (legend=false, xlabel="lambda", xscale=:log10)
    println(size(Xmatrix))
    println(size(yvector))
    coef = GLMNet.coef(cv)
    non_zero_indices = findall(x -> abs(x) > 0, coef)


    y_pred = GLMNet.predict(cv, Xmatrix)
    y_pred_labels = ifelse.(y_pred .>= 1.5, 2, 1)
    println("mean prediction label ", mean(y_pred_labels))
    my_accuracy = mean(y_pred_labels .== yvector)
    println("Accuracy: $my_accuracy, Non-zero indexes: $(length(non_zero_indices))")


    # leafmap!(mask) do mask_node
    #     mask_node.mask.x .= false
    #     return mask_node
    # end

    new_flat_view = ExplainMill.FlatView(mk)
    # new_flat_view[non_zero_indices] = true
    y_pred_inverted = 1 .- y_pred
    for i in 1:length(flat_modification_masks[1])
        mi = new_flat_view.itemmap[i]
        # println(typeof(new_flat_view.masks[mi.maskid].m.stats.present))
        # println(fieldnames(typeof(new_flat_view.masks[mi.maskid].m.stats.present)))
        # println(abs(coef[i]))
        # new_flat_view.masks[mi.maskid].h = abs(coef[i]) == 0 ? 0 : abs(coef[i])
        # if typeof(new_flat_view.masks[mi.maskid].m.stats.present.s) != Array{Float64,1}
        #     new_flat_view.masks[mi.maskid].m.stats.present.s = Float64[]
        # end

        # Append abs(coef[i]) to s
        # push!(new_flat_view.masks[mi.maskid].m.stats.present.s, abs(coef[i]))
        # # Initialize as an empty array if not already an array and append 1
        # if typeof(new_flat_view.masks[mi.maskid].m.stats.present.n) != Array{Int64,1}
        #     new_flat_view.masks[mi.maskid].m.stats.present.n = Int64[]
        # end
        # push!(new_flat_view.masks[mi.maskid].m.stats.present.n, 1)

        # # Initialize as an empty array if not already an array and append 0.0
        # if typeof(new_flat_view.masks[mi.maskid].m.stats.absent.s) != Array{Float64,1}
        #     new_flat_view.masks[mi.maskid].m.stats.absent.s = Float64[]
        # end
        # push!(new_flat_view.masks[mi.maskid].m.stats.absent.s, 0.0)

        # # Initialize as an empty array if not already an array and append 1
        # if typeof(new_flat_view.masks[mi.maskid].m.stats.absent.n) != Array{Int64,1}
        #     new_flat_view.masks[mi.maskid].m.stats.absent.n = Int64[]
        # end
        # push!(new_flat_view.masks[mi.maskid].m.stats.absent.n, 1)
        new_flat_view.masks[mi.maskid].m.stats.present.s[mi.innerid] = abs(coef[i])
        new_flat_view.masks[mi.maskid].m.stats.present.n[mi.innerid] = 1
        new_flat_view.masks[mi.maskid].m.stats.absent.s[mi.innerid] = 0
        new_flat_view.masks[mi.maskid].m.stats.absent.n[mi.innerid] = 1
        # new_flat_view.masks[mi.maskid].stats[mi.innerid] = abs(coef[i]) == 0 ? 0 : abs(coef[i])
        #new_flat_view.masks[mi.maskid].h[mi.innerid] = abs(coef[i])
        # new_flat_view.masks[mi.maskid].x[mi.innerid] = abs(coef[i])
        # new_flat_view.masks[i].h = 0.123
    end

end

function Duff.update!(e::DafExplainer, mk::AbstractVectorMask, o)
    d = simplemask(mk)
    for i in 1:length(d.x)
        !e.banzhaf && !participate(mk)[i] && continue
        Duff.update!(d.stats, o, d.x[i] & participate(mk)[i], i)
    end
end

function StatsBase.sample!(mk::AbstractStructureMask, weights)
    foreach_mask(mk) do m, l
        m .= sample([true, false], weights, length(m))
    end
end