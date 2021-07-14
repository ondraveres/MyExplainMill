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

# function stats(e::GradExplainer, ds, model, i, clustering = ExplainMill._nocluster)

function stats(e::GradExplainer, ds, model)
	mk = create_mask_structure(ds, SimpleMask)
	y = gnntarget(model, ds)
	ps = Flux.Params(map(m -> simplemask(m).x, collectmasks(mk)))
	gs = gradient(() -> Flux.logitcrossentropy(model(ds, mk).data, y), ps)
	mkₕ = mapmask(mk) do m, l
		d = length(m)
		if haskey(gs, m.x)
			HeuristicMask(abs.(gs[m.x])[:])
		else
			HeuristicMask(zeros(Float32, d))
		end
	end
	mkₕ
 end


@deprecate GradExplainer2 GradExplainer
