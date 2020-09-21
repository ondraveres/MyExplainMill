"""
	(i, ii) =  uniquecolumns(x; ϵ = 0)
	i --- indexes of first unique columns in x
"""
function uniquecolumns(x; ϵ = 0)
	ii = fill(0, size(x,2))
	i = Vector{Int}()
	for j in 1:size(x, 2)
		ii[j] != 0 && continue
		_x = x[:,j]
		d = sum(abs.(_x .- x), dims = 1)[:]
		push!(i, j)
		ii[d .<= ϵ] .= length(i)
	end
	i, ii
end

function hclust_fdist(subm, subds, m, ds, dt, ois; stochastic = true, ϵ = 0, δ = 0.05, symmetric = true)
	nobs(subds) == 1 && return([1])
	d = fdist(subm, subds, m, ds, dt, ois; stochastic = stochastic, ϵ = ϵ)
	d = symmetric ? max.(d,d') : d
	cutree(hclust(d, linkage = :ward), h = δ)
end

function hclust_fastfisherdist(subm, subds, m, ds, dt, ois; δ = 0.05)
	nobs(subds) == 1 && return([1])
	d = fastfisherdist(subm, subds, m, ds, dt, ois)
	d = max.(d, d')
	cutree(hclust(d, linkage = :ward), h = δ)
end

function getmetric(name)
	if name == "cosine"
		return((m, ds) -> pairwise(CosineDist(), m(ds).data, dims = 2))
	elseif name == "l2"
		return((m, ds) -> pairwise(Distances.SqEuclidean(), m(ds).data, dims = 2))
	else
		error("unknown metric $(name)")
	end
end

function getclustering(name, δ)
	if name == "hclust"
		return(d -> cutree(hclust(d, linkage = :ward), h = δ))
	elseif name == "dbscan"
		return(d -> dbscan(d, δ, 1).assignments)
	else
		error("unknown clustering $(name)")
	end
end


function getclustering(clustering, metric, δ)
	clustering == "exact" && return((m, ds) ->  uniquecolumns(m(ds).data)[2])
	dfun = getmetric(metric)
	cfun = getclustering(clustering, δ)
	return (model, ds) ->  begin 
		nobs(ds) == 1 && return([1])
		i, ii = uniquecolumns(model(ds).data)
		length(unique(ii)) == 1 && return(fill(1, nobs(ds)))
		d = dfun(model, ds[i])
		d = 0.5 .* (d .+ d')
		cfun(d)[ii]
	end
end


