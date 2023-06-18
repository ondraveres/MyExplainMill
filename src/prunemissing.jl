removable(::Missing) = true
removable(ds::AbstractMillNode) = ismissing(ds.data)
removable(s::Vector{T}) where {T<:AbstractString} = isempty(s)
removable(s::Matrix{T}) where {T<:Number} = false
removable(x::Matrix{Union{Missing, T}}) where {T<:Number} = all(x .=== missing)
removable(x::NGramMatrix) = all(x.S .=== missing)
removable(x::Flux.OneHotMatrix) = all(j == size(x,1) for j in x.indices)
removable(x::Mill.MaybeHotMatrix) = all(x .=== missing)

removemissing(ds::AbstractMillNode) = removable(ds.data) ? missing : ds

function removemissing(ds::BagNode) 
	data = removemissing(ds.data)
	removable(data) ? missing : BagNode(data, ds.bags)
end

function removemissing(ds::ProductNode) 
	ks = [k => removemissing(ds.data[k]) for k in collect(keys(ds.data))]
	ks = filter(k -> !ismissing(k.second), ks)
	isempty(ks) && return(missing)
	dd = (;ks...)
	ProductNode(dd)
end
