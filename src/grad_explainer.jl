#####
#		A hacky immplementation of an explanation using the gradient
#####
using Zygote, Setfield

struct GradExplainer
end

function stats(e::GradExplainer, ds, model, i, n, clustering = ExplainMill._nocluster)
	GradMask(model, ds, model, ds, i)
end

scorefun(e::GradExplainer, x::AbstractStructureMask) = x.mask.stats[:]


function ∇data(m, ds, subm, subds, i)
	mm = replacein(m, subm, identity)
	xx = subm(subds)
	dd = replacein(ds, subds, xx)
	gs = gradient(() -> sum(mm(dd).data[i,:]), Params([xx.data]))
	gs[xx.data]
end

function ∇data(m, ds, subm, xx::Matrix, i)
	gs = gradient(() -> sum(m(ds).data[i,:]), Params([xx]))
	gs[xx]
end

function GradMask(m, ds, subm::BagModel, subds::BagNode, i)
	isnothing(subds.data) && return(EmptyMask())
	nobs(subds.data) == 0 && return(EmptyMask())
	∇x = ∇data(m, ds, subm.im, subds.data, i)
	ms = Mask(size(∇x, 2), d -> sum(abs.(∇x), dims = 1)[:])
	child_mask = GradMask(m, ds, subm.im, subds.data, i)
	BagMask(child_mask, subds.bags, ms)
end

function GradMask(m, ds, subm::ArrayModel, subds::ArrayNode{T,M}, i) where {T<:Flux.OneHotMatrix, M}
	nobs(subds.data) == 0 && return(EmptyMask())
	∇x = ∇data(m, ds, subm, subds, i)
	CategoricalMask(Mask(size(∇x, 2), d -> sum(abs.(∇x), dims = 1)[:]))
end

function GradMask(m, ds, subm::ArrayModel, subds::ArrayNode{T,M}, i) where {T<:Mill.NGramMatrix, M}
	nobs(subds.data) == 0 && return(EmptyMask())
	∇x = ∇data(m, ds, subm, subds, i)
	NGramMatrixMask(Mask(size(∇x, 2), d -> sum(abs.(∇x), dims = 1)[:]))
end

function GradMask(m, ds, subm::ArrayModel, subds::ArrayNode{T,M}, i) where {T<:Matrix, M}
	nobs(subds.data) == 0 && return(EmptyMask())
	∇x = ∇data(m, ds, subm, subds.data, i)
	Mask(subds, d -> sum(abs.(∇x), dims = 2)[:])
end

function GradMask(m, ds, subm::ProductModel, subds::ProductNode, i)
	ks = keys(subds.data)
	s = (;[k => GradMask(m, ds, subm.ms[k], subds.data[k], i) for k in ks]...)
	ProductMask(s)
end


"""
	struct GradExplainer2{T}
		p::T
	end

	Calculates importance of features as an absolute value of gradient with 
	respect to a mask set to all ones multiplying features / adjacency matrix. 
	This explanation was used in GNN explainer by Leskovec et al. in experimental
	section to show that their method is better.

	Disclaimer. The gradients are caclulated by hooking the `diffmask` used in GNN explainer.
	Since weights there are passed through `σ` function, we approximate the correct computation
	by setting mask to invσ(0.999) and then correcting the gradient. By doing so, we are inducing
	about 1e-3 error, but we do not have to add any additional infrastructure.
"""
struct GradExplainer2{T}
	p::T
end

GradExplainer2() = GradExplainer2(0.999f0)

iσ(p) =  log(p) - log(1-p);

function stats(e::GradExplainer2, ds, model, i, clustering = ExplainMill._nocluster)
	soft_model = (ds...) -> softmax(model(ds...));
	mask = ExplainMill.Mask(ds, model, d -> fill(iσ(e.p), d, 1), clustering)
	ms = filter(x -> !isa(x,ExplainMill.AbstractNoMask), collect(NodeIterator(mask)))
	ps = Flux.Params(map(x -> x.mask.stats, ms))
	y = gnntarget(model, ds, i)
	loss() = Flux.logitcrossentropy(model(ds, mask).data, y)
	gs = gradient(loss, ps)
	for p in ps
		p .= abs.(gs[p] ./ gradient(σ, iσ(e.p))[1])
	end
 	mask
 end

scorefun(e::GradExplainer2, x::AbstractStructureMask) = x.mask.stats[:]


struct GradExplainer3{T}
	p::T
end

GradExplainer3() = GradExplainer3(0.999f0)

function stats(e::GradExplainer3, ds, model, i, clustering = ExplainMill._nocluster)
	soft_model = (ds...) -> softmax(model(ds...));
	mask = ExplainMill.Mask(ds, model, d -> fill(iσ(e.p), d, 1), clustering)
	ms = filter(x -> !isa(x,ExplainMill.AbstractNoMask), collect(NodeIterator(mask)))
	ps = Flux.Params(map(x -> x.mask.stats, ms))
	y = gnntarget(model, ds, i)
	loss() = Flux.logitcrossentropy(model(ds, mask).data, y)
	gs = gradient(loss, ps)
	for p in ps
		p .= gs[p] ./ gradient(σ, iσ(e.p))[1]
	end
 	mask
 end

scorefun(e::GradExplainer3, x::AbstractStructureMask) = x.mask.stats[:]
