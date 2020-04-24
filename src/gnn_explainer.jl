using Zygote, HierarchicalUtils

function entropy(x::AbstractArray) 
	x = x[:]
	@assert all(( x.>= 0) .& (x .<=1))
	mask = (x .!= 0) .& (x .!= 1)
	-sum(x[mask] .* log.(x[mask])) / max(1,length(x))
end

_entropy(x::T) where {T<:Real} = (x <=0 || x >= 1) ? zero(T) : - x * log(x)
function _entropy(x::AbstractArray) 
	isempty(x) && return(0f0)
	mapreduce(_entropy, + ,x) / length(x)
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

function stats(model, ds, mask, y, ps)
	lo = Flux.logitcrossentropy(model(ds, mask).data, y)
	x = vcat([σ.(p[:]) for p in ps]...)
	println(" loss: ", lo, " entropy: ", entropy(x), " quantiles: ",map(x -> round(x, digits = 2), quantile(x, 0:0.2:1)))
end

function gnn_explain(ds, model, i, n, clustering;  α = 1f0, β = 5f-3)
	mask = ExplainMill.Mask(ds, model, d -> rand(Float32, d, 1), ExplainMill._nocluster)
	ms = filter(x -> !isa(x,ExplainMill.AbstractNoMask), collect(NodeIterator(mask)))
	ps = Flux.Params(map(x -> x.mask.stats, ms))
	reinit!(ps)
	y = gnntarget(model, ds, i)
	opt = ADAM()
	loss() = Flux.logitcrossentropy(model(ds, mask).data, y)
	println("full sample: ", Flux.logitcrossentropy(model(ds).data, y))
	for step in 1:n
		gs = gradient(loss, ps)
		for p in ps
			gs[p] .+= gradient(x -> regularization(x, α, β), p)[1]
		end
		Flux.Optimise.update!(opt, ps, gs)
		mod(step, 100) == 0 && (print("step: ",step);stats(model, ds, mask, y, ps))
 	end
 	mask
end