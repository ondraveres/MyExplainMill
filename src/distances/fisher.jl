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
		zz.data .= z[:,i]
		y, back = Zygote.pullback(() -> f(pds).data, Flux.params([zz.data]))
		for oi in ois
			sen = Flux.onehotbatch(fill(oi, size(y,2)), 1:size(y,1))
			f∇z = mean(back(sen)[zz.data], dims = 2)
			o[:,:,i] .+= f∇z * f∇z'
		end
	end
end

function fdist(x, ∇x, y)
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
fdist(x, ∇x) = fdist(x, ∇x, x)

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

function fastfisherdist(subm, subds, m, ds, dt, ois; max_thicket_size = 100)
	lens = Mill.findin(ds, subds)
	fastfisherdist(subm, m, subds, lens, dt, ois)
end

function fastfisherdist(submodel, model, subds, lens::Setfield.ComposedLens, dt, ois; max_thicket_size = 100)
	∇z, z = fastfishergrad(model, submodel, subds, lens, dt, ois)
	fdist(z, ∇z)
end

function fastfishergrad(model, subm, subds, lens::Setfield.ComposedLens, dt, ois; max_thicket_size = 100)
	z = subm(subds).data;
	u, ii = uniquecolumns(z)
	@info "$(lens): $(length(u)) unique samples of $(length(ii))"
	o = fastfishergrad(model, subm, z[:,u], lens, dt, ois)
	o[:,:,ii], z
end

function fastfishergrad(model, subm, z::AbstractMatrix, lens::Setfield.ComposedLens, dt, ois; max_thicket_size = 100)
	dt, subdt, baglengths = prunethicket(dt, lens)

	#a little magic to speed things up
	model = replacein(model, subm, IdentityModel())
	zz = subm(subdt);
	dd = replacein(dt, subdt, zz);
	f, pds = Mill.partialeval(model, dd, zz)

	o = similar(z, size(z, 1), size(z)...) .= 0
	Threads.@threads for i in 1:size(z,2)
		zz.data .= z[:,i]
		y, back = Zygote.pullback(() -> f(pds).data, Flux.params([zz.data]))
		for oi in ois
			sen = Flux.onehotbatch(fill(oi, size(y,2)), 1:size(y,1))
			f∇z = meanofmean(back(sen)[zz.data], baglengths)
			o[:,:,i] .+= f∇z * f∇z'
		end
	end
	o
end


"""
	meanofmean(z, baglengths)

	mean of `[z[:,b] for b in baglengths]` where `b` is some bag of lengths in `baglengths`

"""
function meanofmean(z, baglengths)
	s = mapreduce(b -> mean(view(z, :, b), dims = 2), + , Mill.length2bags(baglengths))
	s ./ length(baglengths)
end

"""
	(dt, subdt, n) = prunethicket(dt::Vector, lens::Setfield.ComposedLens; max_thicket_size = 100)

	subset of `dt` of at most size `max_thicket_size` with elements with non-empty `dt[lens]`.
	`n` is sizes of individual parts of `dt[lens]`
"""
function prunethicket(dt, lens::Setfield.ComposedLens; max_thicket_size = 100)
	subdt = map(_dt -> get(_dt, lens), dt)
	ii = findall(map(d -> nobs(d) != 0, subdt))
	isempty(ii) && return(missing)

	ii = length(ii) > max_thicket_size ? sample(ii, max_thicket_size, replace = false) : ii
	dt, subdt = dt[ii], subdt[ii]

	baglengths = map(nobs, subdt)
	dt = reduce(catobs, dt)
	subdt = get(dt, lens)
	(dt, subdt, baglengths)
end