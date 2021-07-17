#####
using Zygote

"""
	struct GradExplainer
	end

	Calculates importance of features as an absolute value of gradient with 
	respect to a mask set to all ones multiplying features / adjacency matrix. 
	This explanation was used in GNN explainer by Leskovec et al. in experimental
	section to show that their method is better.

	The method does not have any parameters, as it is literaly just a calculation of gradients.
"""
struct GradExplainer

end

function stats(e::GradExplainer, ds, model)
	o = softmax(model(ds).data)
	classes = Flux.onecold(o)
	stats(e, ds, model, classes, _nocluster)
end

function stats(e::GradExplainer, ds, model, classes, ::typeof(_nocluster))
	o = softmax(model(ds).data)
	y = Flux.onehotbatch(classes, 1:size(o,1))

	mk = create_mask_structure(ds, d -> SimpleMask(ones(eltype(o), d)))
	ps = Flux.Params(map(m -> simplemask(m).x, collectmasks(mk)))
	gs = gradient(() -> sum(softmax(model(ds, mk).data) .* y), ps)
	mkₕ = mapmask(mk) do m, l
		d = length(m)
		if haskey(gs, m.x)
			HeuristicMask(abs.(gs[m.x])[:])
		else
			HeuristicMask(zeros(eltype(m.x), d))
		end
	end
	mkₕ
 end


@deprecate GradExplainer2 GradExplainer
