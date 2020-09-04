using Flux
using Zygote
using Setfield
using ThreadTools

function removeempty(dt, subdt)
	mask = nobs.(subdt) .!= 0
	dt[mask], subdt[mask]
end
"""
	fishergrad(m, ds, subm::BagModel, subds::BagNode, i::Int)

	fisher gradient of 
"""
fishergrad(m, subm, subds::AbstractNode, dt, subdt::AbstractNode, i::Int; ϵ = 0, stochastic = true) = fishergrad(m, subds, subm, dt, subdt, i:i; ϵ = ϵ, stochastic = stochastic)
fishergrad(m, subm, subds::AbstractNode, dt, subdt::Vector, i::Int; ϵ = 0, stochastic = true) = fishergrad(m, subds, subm, dt, subdt, i:i; ϵ = ϵ, stochastic = stochastic)

function fishergrad(m, subm, subds::AbstractNode, dt::AbstractNode, subdt, ois; ϵ = 0, stochastic = true)
	z = subm(subds).data;
	u, ii = uniquecolumns(z, ϵ = ϵ)
	@info "number of unique samples $(length(u)) from $(length(ii))"
	o = fishergrad(m, subm, z[:,u], dt, subdt, ois, stochastic = stochastic)
	o[:,:,ii]
end

function fishergrad(m, subm, subds::AbstractNode, dt::Vector, subdt, ois; ϵ = 0, stochastic = true)
	dt, subdt = removeempty(dt, subdt)
	z = subm(subds).data;
	if isempty(dt) || isempty(subdt)
		@info "thicket is empty, cannot estimate fisher information matrix"
		return(zero(z))
	end
	u, ii = uniquecolumns(z, ϵ = ϵ)
	@info "number of unique samples $(length(u)) from $(length(ii))"
	os = tmap(i -> fishergrad(m, subm, z[:,u], dt[i], subdt[i], ois, stochastic = stochastic), 1:length(dt))
	o = reduce(+, os) ./ length(dt)
	o[:,:,ii]
end

function fishergrad(m, subm, z::AbstractMatrix, dt, subdt, ois; stochastic = true)
	mm = replacein(m, subm, IdentityModel())
	d, l = size(z)
	o = similar(z, d, d, l) 
	o .= 0
	_fishergrad!(o, z, mm, ois, subm, dt, subdt, stochastic)
	o
end

function _fishergrad!(o, z, mm, ois, subm, dt::Vector, subdt, stochastic)
	for j in 1:length(dt)
		_fishergrad!(o, z, mm, i, subm, dt[j], subdt[j], stochastic)
	end
end

function _fishergrad!(o, z, mm, ois, subm, dt::Mill.AbstractNode, subdt, stochastic)
	@assert nobs(dt) == 1
	zz = subm(subdt);
	dd = replacein(dt, subdt, zz);
	f, pds = Mill.partialeval(mm, dd, zz)

	for i in 1:size(z,2)
		zz.data[:,1] .= z[:,i]
		y, back = Zygote.pullback(() -> f(pds).data, Flux.params([zz.data]))
		sen = deepcopy(y) .= 0
		for oi in ois
			sen[oi] = 1
			f∇z = back(sen)[zz.data][:,1]
			o[:,:,i] .+= f∇z * f∇z'
			sen[oi] = 0 
		end
	end
end

function fdist(x::AbstractMatrix, ∇x::AbstractArray{T,3}) where {T}
	d = similar(x, size(x, 2), size(x, 2))
	for i in 1:size(x,2)
		d[i,i] = 0
		for j in 1:size(x,2)
			δ = view(x, :, i) - view(x, :, j)
			d[i,j] = δ' * view(∇x, :, :, i) * δ
		end
	end
	d
end


function fdist(x::AbstractMatrix, ∇x::AbstractArray{T,3}, y::AbstractMatrix) where {T}
	d = similar(x, size(x, 2), size(y, 2))
	for i in 1:size(x,2)
		d[i,i] = 0
		for j in 1:size(y,2)
			δ = view(x, :, i) - view(y, :, j)
			d[i,j] = δ' * view(∇x, :, :, i) * δ
		end
	end
	d
end

function fdist(subm, subds, m, ds, dt, ois; stochastic = true, ϵ = 0)
	lens = Mill.findin(ds, subds)
	fdist(subm, m, subds, lens, dt, ois; stochastic = stochastic, ϵ = ϵ)
end

function fdist(submodel, model, subds, lens::Setfield.ComposedLens, dt, ois; stochastic = true, ϵ = 0)
	subdt = map(_dt -> get(_dt, lens), dt)
	ii = map(d -> nobs(d) != 0, subdt)
	dt, subdt = dt[ii], subdt[ii]
	isempty(subdt) && return(pairwise(SqEuclidean(), submodel(subds).data, dims = 2))
	if length(subdt) > 100 
		ii = sortperm(subdt, lt = (x,y) -> nobs(x) < nobs(y)) 
		ii = ii[1:100]
		dt, subdt = dt[ii], subdt[ii]
	end
	@info "length of thicket $(length(dt))"
	∇z = fishergrad(model, submodel, subds, dt, subdt, ois, stochastic = true)
	z = submodel(subds).data
	fdist(z, ∇z)
end



