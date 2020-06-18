function uniquecolumns(x; ϵ = 0)
	allidxs = fill(0, size(x,2))
	u = Vector{Int}()
	for i in 1:size(x, 2)
		allidxs[i] != 0 && continue
		_x = x[:,i]
		d = sum(abs.(_x .- x), dims = 1)[:]
		allidxs[d .<= ϵ] .= i
		push!(u, i)
	end
	u, allidxs
end

function addtobags(xx, bags, z)
	nx = hcat(xx, z)
	e = size(nx,2)
	nbags = Mill.ScatteredBags([isempty(b) ? Vector{Int}() : push!(collect(b), e) for b in ds])
	BagNode(nx, nbags)
end

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
	o = fishergrad(m, subm, z[:,u], dt, subdt, ois, stochastic = stochastic)
	o[:,:,ii]
end

function fishergrad(m, subm, subds::AbstractNode, dt::Vector, subdt, ois; ϵ = 0, stochastic = true)
	dt, subdt = removeempty(dt, subdt)
	z = subm(subds).data;
	u, ii = uniquecolumns(z, ϵ = ϵ)
	os = tmap1(i -> fishergrad(m, subm, z[:,u], dt[i], subdt[i], ois, stochastic = stochastic), 1:length(dt))
	o = reduce(+, os) ./ length(os)
	o[:,:,ii]
end

function fishergrad(m, subm, z::AbstractMatrix, dt, subdt, ois; stochastic = true)
	mm = replacein(m, subm, identity);
	d, l = size(z)
	o = similar(z, d, d, l) 
	n = _fishergrad!(o, z, mm, ois, subm, dt, subdt, stochastic)
	o ./= n
end


function _fishergrad!(o, z, mm, ois, subm, dt::Vector, subdt, stochastic)
	n = 0 
	for j in 1:length(dt)
		n += _fishergrad!(o, z, mm, i, subm, dt[j], subdt[j], stochastic)
	end
	n
end

# function _fishergrad!(o, z, mm, ois, subm, dt::Mill.AbstractNode, subdt, stochastic)
# 	@assert nobs(dt) == 1
# 	zz = subm(subdt);
# 	dd = replacein(dt, subdt, zz);
# 	f, pds = Mill.partialeval(mm, dd, zz)
# 	o .= 0
# 	n = 1
# 	js = stochastic ? [rand(1:nobs(zz))] : collect(1:nobs(zz))
# 	for j in js
# 		t = zz.data[:,j]
# 		for i in 1:size(z,2)
# 			zz.data[:,j] .= z[:,i]
# 			for oi in ois
# 				gs = gradient(() -> f(pds).data[oi], Params([zz.data]));
# 				_o = gs[zz.data][:,j]
# 				o[:,:,i] .+= _o * _o'
# 			end
# 			n += 1
# 		end
# 		zz.data[:,j] .= t
# 	end
# 	n
# end

function _fishergrad!(o, z, mm, ois, subm, dt::Mill.AbstractNode, subdt, stochastic)
	@assert nobs(dt) == 1
	zz = subm(subdt);
	dd = replacein(dt, subdt, zz);
	f, pds = Mill.partialeval(mm, dd, zz)
	o .= 0
	n = 1
	js = stochastic ? [rand(1:nobs(zz))] : collect(1:nobs(zz))
	for j in js
		t = zz.data[:,j]
		for i in 1:size(z,2)
			zz.data[:,j] .= z[:,i]
			y, back = pullback(() -> f(pds).data, Params([zz.data]))
			sen = deepcopy(y) .= 0
			for oi in ois
				sen[i] = 1
				gs = back(sen)
				_o = gs[zz.data][:,j]
				o[:,:,i] .+= _o * _o'
				sen[i] = 0 
			end
			n += 1
		end
		zz.data[:,j] .= t
	end
	o ./= length(ois)
	n
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


