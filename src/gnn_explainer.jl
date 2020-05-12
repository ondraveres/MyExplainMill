using Zygote, HierarchicalUtils, Setfield

struct GnnExplainer
	α::Float32
	β::Float32
end

GnnExplainer() = GnnExplainer(1f0, 5f-3)

function stats(e::GnnExplainer, ds, model, i, n, clustering = ExplainMill._nocluster; threshold = 0.1)
	soft_model = (ds...) -> softmax(model(ds...));
	mask = ExplainMill.Mask(ds, model, d -> rand(Float32, d, 1), clustering)
	ms = filter(x -> !isa(x,ExplainMill.AbstractNoMask), collect(NodeIterator(mask)))
	ps = Flux.Params(map(x -> x.mask.stats, ms))
	reinit!(ps)
	y = gnntarget(model, ds, i)
	opt = ADAM()
	loss() = Flux.logitcrossentropy(model(ds, mask).data, y)
	println("logitcrossentropy sample: ", Flux.logitcrossentropy(model(ds).data, y))
	@timeit to "optimizing mask" for step in 1:n
		gs = gradient(loss, ps)
		for p in ps
			gs[p] .+= gradient(x -> regularization(x, e.α, e.β), p)[1]
		end
		Flux.Optimise.update!(opt, ps, gs)
		# mod(step, 100) == 0 && (print("step: ",step);iteration_stats(model, ds, mask, y, ps))
 	end
 	print("step: ",n); iteration_stats(model, ds, mask, y, ps)
 	mask
end

scorefun(e::GnnExplainer, x::AbstractExplainMask) = σ.(x.mask.stats[:])

function regularization(p::AbstractArray{T}, α, β) where {T<:Real} 
	x = σ.(p)
	α * entropy(x) + β * sum(x)
end

function gnntarget(model, ds, i)
	d, l = size(model(ds).data)
	y = Flux.onehotbatch(fill(i, l), 1:d)
end

function reinit!(ps, mode = :random)
	num_nodes = sum(length(p) for p in ps)
	s = √(1 / num_nodes)
	for p in ps 
		p .= mode == :random ? randn(size(p)) .* s : 1
	end
end

function iteration_stats(model, ds, mask, y, ps)
	lo = Flux.logitcrossentropy(model(ds, mask).data, y)
	x = vcat([σ.(p[:]) for p in ps]...)
	println(" loss: ", lo, " entropy: ", entropy(x), " quantiles: ",map(x -> round(x, digits = 2), quantile(x, 0:0.2:1)))
end