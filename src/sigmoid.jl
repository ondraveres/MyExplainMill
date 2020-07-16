using LinearAlgebra
invσ(x::Real) = log(x)  - log(1 - x) 

function rescale(c_min, c_max, n_min, n_max)
	δ = c_max - c_min
	δ[δ .== 0] .= 1
	W = (n_max - n_min) ./ δ
	b = n_min - c_min .* W
	Diagonal(W), b
end

function minimax(x)
	minimum(x, dims = 2)[:], maximum(x, dims = 2)[:]
end

minimax(x::ArrayNode) = minimax(x.data)

fuseaffine!(d, c_min, c_max, n_min, n_max) = fuseaffine!(d, rescale(c_min, c_max, n_min, n_max)...)
fuseaffine!(d::Dense, W::Vector, b::Vector) = fuseaffine!(d, Diagonal(W), b)
fuseaffine!(d::Dense, ds::Dense) = fuseaffine!(d, ds.W, ds.b)
function fuseaffine!(d::Dense, W, b)
	d.b .+= d.W * b 
	d.W .= d.W * W
end


fuseaffine!(d::Chain, mnmnx...) = fuseaffine!(d[1], mnmnx...)
fuseaffine!(d::ArrayModel, mnmnx...) = fuseaffine!(d.m, mnmnx...)

function fuseaffine!(d::T, mnmnx...) where {T}
	@error "unsupported fuseaffine! for $(T)"
end

function scale201(w, b, x)
	xmin, xmax = minimax(w*x .+ b)
	δ = xmax .- xmin
	δ[δ .== 0] .= 1
	b = (b .- xmin) ./ δ
	w = w ./ δ
	(w, b)
end

"""
	sigmoid(d::Dense, α,  x)

change the transfer function to `σ` and weights such that the output
is similar, i.e it operates in the linear part or in the elbow 
"""
function sigmoid(d::Dense{F,A,B}, α,  xmin, xmax) where {F<:typeof(NNlib.relu), A, B}
	δ = xmax
	δ[δ .== 0] .= 1
	w = d.W ./ δ
	b = d.b ./ δ
	b .+= invσ.(α)
	w .*= invσ(1 - α) - invσ(α)
	Dense(w, b, σ)
end

function sigmoid(d::Dense{F,A,B}, α,  xmin, xmax) where {F<:typeof(identity), A, B}
	δ = xmax .- xmin
	δ[δ .== 0] .= 1
	b = (d.b .- xmin) ./ δ
	w = d.W ./ δ
	w .*= invσ(1 - α) - invσ(α)
	b .*= invσ(1 - α) - invσ(α)
	b .+= invσ.(α)
	Dense(w, b, σ)
end

function sigmoid(d::Dense, α, x)
	sigmoid(d, α, minimax(d(x))...)
end

function sigmoid(d, α, xmin, xmax)
	@warn "sigmoid for $(typeof(d)) not implemented"
	d
end

function sigmoid(d, α, x)
	@warn "sigmoid for $(typeof(d)) not implemented"
	d
end

function sigmoid(d::Chain, α,  x)
	length(d) == 1 && return(sigmoid(d.layers[1], α, x))
	sigmoid(d, α, minimax(d(x))...)
end

function sigmoid(d::Chain, α, xmin, xmax)
	ds = sigmoid(d[end], α, xmin, xmax)
	Chain(d[1:end - 1].layers..., ds)
end

function sigmoid(m::ArrayModel, α, x::ArrayNode)
	mσ = sigmoid(m.m, α, x.data)
	ArrayModel(mσ)
end

function sigmoid(m::BagModel, α, x::BagNode)
	ismissing(x.data) && return(m)
	n_xmin, n_xmax = minimax(m.a(m.im(x.data), x.bags))
	imσ = sigmoid(m.im, α, x.data)
	xx = m.a(imσ(x.data), x.bags)
	c_xmin, c_xmax = minimax(xx)
	fuseaffine!(m.bm, c_xmin, c_xmax, n_xmin, n_xmax)
	bmσ = sigmoid(m.bm, α, xx)
	BagModel(imσ, m.a, bmσ)
end

function sigmoid(m::ProductModel, α, x::ProductNode)
	n_xmin, n_xmax = minimax(ProductModel(m.ms, identity)(x))
	msσ = map(k -> k => sigmoid(m.ms[k], α, x.data[k]), keys(m.ms))
	msσ = (;msσ...)
	xx = ProductModel(msσ, identity)(x)
	c_xmin, c_xmax = minimax(xx)
	fuseaffine!(m.m, c_xmin, c_xmax, n_xmin, n_xmax)
	mσ = sigmoid(m.m, α, xx)
	ProductModel(msσ, mσ)
end

function sigmoid(m, α, x, skip::IdDict)
	haskey(skip, m) && return(m)
	return(sigmoid(m, α, x))
end
