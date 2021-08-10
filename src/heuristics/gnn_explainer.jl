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
prunemask(m::GnnMask) = σ.(m.x) .> 0
simplemask(m::GnnMask) = m

function stats(e::GnnExplainer, ds, model, classes = onecold(model, ds), clustering = _nocluster)
	y = gnntarget(model, ds, classes)
	f(o) = Flux.logitcrossentropy(o, y)
	statsf(e, ds, model, f, clustering)
end

"""
	statsf(e::GnnExplainer, ds, model, f, clustering::typeof(_nocluster))

	A functional api, where `f` is a function used to calculate the heuristic. You
	should very well know, what you are doing here
"""
function statsf(e::GnnExplainer, ds, model, f, clustering::typeof(_nocluster))
	mk = create_mask_structure(ds, GnnMask)
	ps = Flux.Params(map(m -> simplemask(m).x, collectmasks(mk)))
	reinit!(ps)
	opt = ADAM(0.01, (0.5, 0.999))
	for step in 1:e.n
		gs = gradient(() -> f(model(ds, mk).data), ps)
		for p in ps
			if haskey(gs, p)
				gs[p] .+= gradient(x -> regularization(x, e.α, e.β), p)[1]
				gs.grads[p] = reshape(gs[p], size(p))	#an awful fix to Zygote bug
			end
		end
		Flux.Optimise.update!(opt, ps, gs)
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

function gnntarget(model, ds, classes::Vector{Int})
	d = size(model(ds).data, 1)
	y = Flux.onehotbatch(classes, 1:d)
end

function gnntarget(model, ds, class::Int)
	gnntarget(model, ds, fill(class, nobs(ds)))
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


######
#	an implementation of an entropy which does not produce NaNs even in gradients
######
function entropy(x::AbstractArray) 
	x = x[:]
	@assert all(( x.>= 0) .& (x .<=1))
	mask = (x .!= 0) .& (x .!= 1)
	-sum(x[mask] .* log.(x[mask])) / max(1,length(x))
end

_entropy(x::T) where {T<:Real} = (x <=0 || x >= 1) ? zero(T) : - x * log(x)
function _entropy(x::AbstractArray) 
	isempty(x) && return(0f0)
	mapreduce(_entropy, + , x) / length(x)
end

_∇entropy(Δ, x::T) where {T<:Real} = (x <=0 || x >= 1) ? zero(T) : - Δ * (one(T) +  log(x))

function _∇entropy(Δ, x)
	Δ /= length(x)
	(_∇entropy.(Δ, x), )
end

Zygote.@adjoint function entropy(x::AbstractArray) 
	@assert all(( x.>= 0) .& (x .<=1))
	return(_entropy(x), 
		Δ -> _∇entropy(Δ, x))
end

