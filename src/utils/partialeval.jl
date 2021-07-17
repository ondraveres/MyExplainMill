import Mill: partialeval

function Mill.partialeval(model::M, ds::ArrayNode, mk, masks) where {M<:Union{IdentityModel, ArrayModel}}
	mk ∈ masks && return(model, ds, mk, true)
	mk.mask ∈ masks && return(model, ds, mk, true)
	return(ArrayModel(identity), model(ds), EmptyMask(), false)
end

function Mill.partialeval(model::LazyModel{N}, ds::LazyNode{N}, mk, masks) where {N}
	mk ∈ masks && return(model, ds, mk, true)
	mk.mask ∈ masks && return(model, ds, mk, true)
	return(ArrayModel(identity), model(ds), EmptyMask(), false)
end

function Mill.partialeval(model::BagModel, ds::BagNode, mk::BagMask, masks)
	im, ids, childms, keep = Mill.partialeval(model.im, ds.data, mk.child, masks)
	if (mk ∈ masks) | (mk.mask ∈ masks) | keep
		return(BagModel(im, model.a,  model.bm), BagNode(ids, ds.bags, ds.metadata), BagMask(childms, mk.bags, mk.mask), true)
	end
	return(ArrayModel(identity), model.bm(model.a(ids, ds.bags)), EmptyMask(), false)
end

function Mill.partialeval(model::BagModel, ds::BagNode, mk::EmptyMask, masks)
	return(ArrayModel(identity), model(ds), EmptyMask(), false)
end

function Mill.partialeval(model::ProductModel{MS,M}, ds::ProductNode{P,T}, mk::ProductMask, masks) where {P<:NamedTuple,T,MS<:NamedTuple, M} 
	ks = keys(model.ms)
	mods = map(ks) do k
		Mill.partialeval(model.ms[k], ds.data[k], mk[k], masks)
	end
	childmodels = map(f -> f[1], mods)
	childds = map(f -> f[2], mods)
	childms = map(f -> f[3], mods)
	if any(f[4] for f in mods)
		return(ProductModel((;zip(ks, childmodels)...), model.m), ProductNode((;zip(ks, childds)...), ds.metadata), ProductMask((;zip(ks, childms)...)), true)
	end
	return(ArrayModel(identity), model.m(vcat(childds...)), EmptyMask(), false)

end

function Mill.partialeval(model::ProductModel, ds::ProductNode, mk::EmptyMask, masks)
	return(ArrayModel(identity), model.m(vcat(childds...)), EmptyMask(), false)
end