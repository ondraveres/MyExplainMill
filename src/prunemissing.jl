Base.ismissing(ds::Mill.AbstractNode) = ismissing(ds.data)

prunemissing(ds::Mill.AbstractNode) = ismissing(ds.data) ? missing : ds
function prunemissing(ds::Mill.BagNode) 
	data = prunemissing(ds.data)
	ismissing(data) ? missing : BagNode(data, ds.bags)
end

function prunemissing(ds::Mill.ProductNode) 
	ks = filter(k -> !ismissing(ds.data[k]), collect(keys(ds.data)))
	isempty(ks) && return(missing)
	ProductNode((;[k => prunemissing(ds.data[k]) for k in ks]...))
end

