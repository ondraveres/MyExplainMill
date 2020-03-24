Base.ismissing(ds::Mill.AbstractNode) = ismissing(ds.data)

removemissing(ds::Mill.AbstractNode) = ismissing(ds.data) ? missing : ds
function removemissing(ds::Mill.BagNode) 
	data = removemissing(ds.data)
	ismissing(data) ? missing : BagNode(data, ds.bags)
end

function removemissing(ds::Mill.ProductNode) 
	ks = filter(k -> !ismissing(ds.data[k]), collect(keys(ds.data)))
	isempty(ks) && return(missing)
	ProductNode((;[k => removemissing(ds.data[k]) for k in ks]...))
end

