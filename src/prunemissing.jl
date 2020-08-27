removable(::Missing) = true
# removable(::Missing) = true
removable(ds::Mill.AbstractNode) = ismissing(ds.data)
removable(s::Vector{T}) where {T<:AbstractString} = isempty(s)
removable(s::Matrix{T}) where {T<:Number} = false
removable(x::NGramMatrix{String}) = all(isempty.(x.s))
removable(x::Flux.OneHotMatrix{Array{Flux.OneHotVector,1}}) = all(j.ix == x.height for j in x.data)

removemissing(ds::Mill.AbstractNode) = removable(ds.data) ? missing : ds

function removemissing(ds::Mill.BagNode) 
	data = removemissing(ds.data)
	removable(data) ? missing : BagNode(data, ds.bags)
end

function removemissing(ds::Mill.ProductNode) 
	ks = [k => removemissing(ds.data[k]) for k in collect(keys(ds.data))]
	ks = filter(k -> !ismissing(k.second), ks)
	isempty(ks) && return(missing)
	ProductNode((;ks...))
end

@deprecate prunemissing removemissing