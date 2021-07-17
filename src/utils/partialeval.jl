import Mill: partialeval

function Mill.partialeval(model::M, ds::ArrayNode, ms, masks) where {M<:Union{IdentityModel, ArrayModel}}
	ms ∈ masks && return(model, ds, ms, true)
	return(ArrayModel(identity), model(ds), EmptyMask(), false)
end

function Mill.partialeval(model::LazyModel{N}, ds::LazyNode{N}, ms, masks) where {N}
	ms ∈ masks && return(model, ds, ms, true)
	return(ArrayModel(identity), model(ds), EmptyMask(), false)
end

function Mill.partialeval(model::BagModel, ds::BagNode, ms::BagMask, masks)
	im, ids, childms, keep = Mill.partialeval(model.im, ds.data, ms.child, masks)
	if (ms ∈ masks) | keep
		return(BagModel(im, model.a,  model.bm), BagNode(ids, ds.bags, ds.metadata), BagMask(childms, ms.bags, ms.mask), true)
	end
	return(ArrayModel(identity), model.bm(model.a(ids, ds.bags)), EmptyMask(), false)
end

function Mill.partialeval(model::BagModel, ds::BagNode, ms::EmptyMask, masks)
	return(ArrayModel(identity), model(ds), EmptyMask(), false)
end

function Mill.partialeval(model::ProductModel{MS,M}, ds::ProductNode{P,T}, ms::ProductMask, masks) where {P<:NamedTuple,T,MS<:NamedTuple, M} 
	ks = keys(model.ms)
	mods = map(ks) do k
		Mill.partialeval(model.ms[k], ds.data[k], ms[k], masks)
	end
	childmodels = map(f -> f[1], mods)
	childds = map(f -> f[2], mods)
	childms = map(f -> f[3], mods)
	if any(f[4] for f in mods)
		return(ProductModel((;zip(ks, childmodels)...), model.m), ProductNode((;zip(ks, childds)...), ds.metadata), ProductMask((;zip(ks, childms)...)), true)
	end
	return(ArrayModel(identity), model.m(vcat(childds...)), EmptyMask(), false)

end

function Mill.partialeval(model::ProductModel{MS,M}, ds::ProductNode{P,T}, ms::ProductMask, masks) where {P<:Tuple,T,MS<:Tuple, M} 
	mods = map(1:length(m.ms)) do k
		Mill.partialeval(model.ms[k], ds.data[k], ms[k], masks)
	end
	childmodels = map(f -> f[1], mods)
	childds = map(f -> f[2], mods)
	childms = map(f -> f[3], mods)
	if any(f[4] for f in mods)
		return(ProductModel(tuple(childmodels...), model.m), ProductNode(tuple(childds...), ds.metadata), ProductMask(tuple(childms...)), true)
	end
	return(ArrayModel(identity), model.m(vcat(childds...)), EmptyMask(), false)

end

function Mill.partialeval(model::ProductModel, ds::ProductNode, ms::EmptyMask, masks)
	return(ArrayModel(identity), model.m(vcat(childds...)), EmptyMask(), false)
end