using Zygote, Setfield

struct GnnExplainer
	n::Int
	α::Float32
	β::Float32
end

GnnExplainer(n::Int) = GnnExplainer(n, 1f0, 5f-3)
GnnExplainer(n) = GnnExplainer(200)


"""
	struct GnnMask{T} <: AbstractVectorMask
		x::Vector{T}
	end

	this is a very stripped down version of masks implementing just the 
	bare minimum needed to calculate the statistics for gnn explainer. 
	Especially notice the `σ` used in `diffmask`. For safety, it does 
	not implement what is not needed.
"""
struct GnnMask{T} <: AbstractVectorMask
	x::Vector{T}
end

GnnMask(d::Int) = GnnMask(ones(Float32, d))
diffmask(m::GnnMask) = σ.(m.x)
simplemask(m::GnnMask) = m

# function stats(e::GnnExplainer, ds, model, i, clustering = ExplainMill._nocluster)
function stats(e::GnnExplainer, ds, model)
	mk = create_mask_structure(ds, GnnMask)
	y = gnntarget(model, ds)
	ps = Flux.Params(map(m -> simplemask(m).x, collectmasks(mk)))
	reinit!(ps)
	opt = ADAM(0.01, (0.5, 0.999))
	loss() = Flux.logitcrossentropy(model(ds, mk).data, y)
	# println("logitcrossentropy sample: ", Flux.logitcrossentropy(model(ds).data, y))
	for step in 1:e.n
		gs = gradient(loss, ps)
		for p in ps
			if haskey(gs, p)
				gs[p] .+= gradient(x -> regularization(x, e.α, e.β), p)[1]
				gs.grads[p] = reshape(gs[p], size(p))	#an awful fix to Zygote bug
			end
		end
		Flux.Optimise.update!(opt, ps, gs)
		# iteration_stats(model, ds, mk, y, ps)
 	end

	mkₕ = mapmask(mk) do m, l
		HeuristicMask(σ.(m.x)[:])
	end
	mkₕ
end

function regularization(p::AbstractArray{T}, α, β) where {T<:Real} 
	x = σ.(p)
	α * entropy(x) + β * sum(x)
end

function gnntarget(model, ds, i)
	d, l = size(model(ds).data)
	y = Flux.onehotbatch(fill(i, l), 1:d)
end

function gnntarget(model, ds)
	o = softmax(model(ds).data)
	d, l = size(o)
	Flux.onehotbatch(Flux.onecold(o), 1:d)
end

function reinit!(ps, mode = :random)
	num_nodes = mapreduce(length, +, ps)
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
