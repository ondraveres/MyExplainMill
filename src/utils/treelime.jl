using JLD2
using ExplainMill
using Distributions

@enum Direction begin
    UP
    DOWN
end

@enum LimeType begin
    FLAT
    LAYERED
end

@enum Distance begin
    CONST
    JSONDIFF
end

struct TreeLimeExplainer
    n::Int
    rel_tol::Float64
    type::LimeType
    direction::Direction
    perturbation_chance::Float64
    distance::Distance
end
# include("setops.jl")

function treelime(e::TreeLimeExplainer, ds::AbstractMillNode, model::AbstractMillModel, extractor)
    mk = ExplainMill.create_mask_structure(ds, d -> ExplainMill.ParticipationTracker(ExplainMill.SimpleMask(fill(true, d))))
    treelime!(e, mk, ds, model, extractor)
    return mk
end

function treelime!(e::TreeLimeExplainer, mk::ExplainMill.AbstractStructureMask, ds, model, extractor)
    og_class = Flux.onecold((model(ds)))[1]
    min_cg = e.rel_tol * ExplainMill.logitconfgap(model, ds[mk], og_class)[1]
    globat_flat_view = ExplainMill.FlatView(mk)
    max_depth = maximum(item.level for item in globat_flat_view.itemmap)
    items_ids_at_level = []
    mask_ids_at_level = []
    for depth in 1:max_depth
        current_depth_itemmap = filter(mask -> mask.level == depth, globat_flat_view.itemmap)
        current_depth_item_ids = [item.itemid for item in current_depth_itemmap]
        current_depth_mask_ids = unique([item.maskid for item in current_depth_itemmap])

        push!(items_ids_at_level, current_depth_item_ids)
        push!(mask_ids_at_level, current_depth_mask_ids)
    end
    for i in 1:length(globat_flat_view.itemmap)
        globat_flat_view[i] = 1
    end
    ##1 keeps, 0 deletes
    layers = collect(1:max_depth)
    if e.direction == UP
        layers = reverse(layers)
    end

    if e.type == FLAT
        layers = [1]
    end

    for layer in layers
        flat_modification_masks = []
        labels = []
        distances = []
        og_mk = ExplainMill.create_mask_structure(ds, d -> SimpleMask(d))
        og = ExplainMill.e2boolean(ds, og_mk, extractor)

        n = nothing
        if e.type == FLAT
            n = e.n
        else
            n = round(Int, e.n / max_depth)
        end


        for _ in 1:n
            perturbation_chance = nothing
            if e.perturbation_chance == 0
                perturbation_chance = rand()
            else

                # Create a truncated normal distribution with mean 0.1, standard deviation 0.1, min 0, and max 1
                d = Truncated(Normal(0.0, e.perturbation_chance), 0, 1)

                # Generate a random number from the distribution
                rand(d)
                perturbation_chance = rand(d)
            end
            if e.type == FLAT
                full_sample!(mk, Weights([1 - perturbation_chance, perturbation_chance]))
            else
                sample_at_level!(mk, Weights([1 - perturbation_chance, perturbation_chance]), layer)
            end

            updateparticipation!(mk)
            local_flat_view = ExplainMill.FlatView(mk)
            p_flat_view = ExplainMill.participate(ExplainMill.FlatView(mk))
            # for i in 1:length(p_flat_view)
            #     if flat_view[i] && p_flat_view[i]
            #         println("MATCH")
            #     else
            #         println("Not match")
            #     end
            # end
            new_mask_bool_vector = [(p_flat_view[i] && local_flat_view[i]) for i in 1:length(local_flat_view.itemmap)]


            current_level = vcat((mask.m.x for mask in local_flat_view.masks[mask_ids_at_level[layer]])...)

            if e.type == FLAT
                push!(flat_modification_masks, new_mask_bool_vector)
            else
                push!(flat_modification_masks, current_level)
            end


            push!(labels, argmax(model(ds[mk]))[1])
            # println(argmax(model(ds[mk]))[1])




            # s = ExplainMill.e2boolean(ds, mk, extractor)
            if e.distance == JSONDIFF
                # println(nnodes(s))
                # println(nleaves(s))
                ce = jsondiff(og, s)
                ec = jsondiff(s, og)
                push!(distances, nleaves(ce) + nleaves(ec))
            end

        end
        # println(length(flat_modification_masks[1]), flat_modification_masks[1])
        # println("labels", labels)


        # println("og_class", og_class)
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
        # println("weights are", weights .* normalized_distances)
        if e.distance == JSONDIFF
            weights = weights .* (1 ./ ((distances .+ 1e-6)))
        end

        # println(typeof(Xmatrix))
        # println(typeof(yvector))

        # println(Xmatrix)

        @save "Xmatrix-$(e.type)-$(layer).jld2" Xmatrix

        cgs = []
        non_zero_lengths = []
        nleaves_list = []

        lambdas = collect(range(0.000, stop=1, step=0.0005))
        mylambdas = lambdas
        pushfirst!(lambdas, 0.0)
        path = nothing
        try
            path = glmnet(Xmatrix, yvector; weights=weights, alpha=1.0, lambda=lambdas)#nlambda=1000)#, lambda=[lambda])
        catch
            "ZERO VARIANCE USING ONES"
        end
        lambdas = []
        betas = nothing
        try
            betas = convert(Matrix, path.betas)
        catch
            betas = ones(size(Xmatrix, 2), length(mylambdas))
        end
        for i in 1:length(mylambdas)
            coef = betas[:, i]
            if i == 1
                println("######")
                println("lambda ", mylambdas[i])
                coef .+= 0.1
                betas[:, i] .+= 0.1
                non_zero_indices = findall(x -> abs(x) > 0, coef)
                println("non_zero_indices ration ", length(non_zero_indices) / length(coef))
            end

            non_zero_indices = findall(x -> abs(x) > 0, coef)
            if e.type == FLAT
                for i in 1:length(globat_flat_view.itemmap)
                    globat_flat_view[i] = 0
                end
                for i in non_zero_indices
                    globat_flat_view[i] = 1
                end
            else
                for i in items_ids_at_level[layer]
                    globat_flat_view[i] = 0
                end

                for i in items_ids_at_level[layer][non_zero_indices]
                    globat_flat_view[i] = 1
                end
            end

            cg = ExplainMill.logitconfgap(model, ds[mk], og_class)[1]
            push!(lambdas, mylambdas[i])
            push!(non_zero_lengths, length(non_zero_indices))
            push!(cgs, cg)
            s = ExplainMill.e2boolean(ds, mk, extractor)
            push!(nleaves_list, nleaves(s))
            # printtree(mk)
            if i == 1
                println("cg: ", cg)
                println("######")
            end
            if length(non_zero_indices) == 0
                break
            end
        end
        positive_indices = findall(x -> x >= min_cg, cgs)
        # println("cgs", cgs)
        # println("min_cg", min_cg)

        if length(positive_indices) > 0
            # println("non_zero_lengths", non_zero_lengths)
            # println("positive_indices", positive_indices)
            best_index = findmin(non_zero_lengths[positive_indices])[2]
            # println("best_index", best_index)
            coef = betas[:, positive_indices[best_index]]
            # println("positive_indices[best_index]", positive_indices[best_index])
            # println("coef", coef)
            non_zero_indices = findall(x -> abs(x) > 0, coef)
            if e.type == FLAT
                for i in 1:length(globat_flat_view.itemmap)
                    globat_flat_view[i] = 0
                end
                for i in non_zero_indices
                    globat_flat_view[i] = 1
                end
            else
                for i in items_ids_at_level[layer]
                    globat_flat_view[i] = 0
                end
                for i in items_ids_at_level[layer][non_zero_indices]
                    globat_flat_view[i] = 1
                end
            end
        end


        cg = ExplainMill.logitconfgap(model, ds[mk], og_class)[1]


        println("END OF LAYER $(layer) CG: ", cg)

        # println(e.rel_tol)
        rt = e.rel_tol

        @save "visual/cg_lambda_plot_$(e.n)_$(e.type)_$(e.direction)_$(layer)_$(e.rel_tol)_$(og_class).jld2" lambdas cgs non_zero_lengths nleaves_list og_class min_cg rt
    end
end

function sample_at_level!(mk::ExplainMill.AbstractStructureMask, weights, level)
    ExplainMill.foreach_mask(mk) do m, l
        if l == level
            m.m.x .= sample([true, false], weights, length(m))
        end
    end
end

function full_sample!(mk::ExplainMill.AbstractStructureMask, weights)
    foreach_mask(mk) do m, l
        m.m.x .= sample([true, false], weights, length(m))
    end
end
