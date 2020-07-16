struct Ensemble{M,O} <: AbstractMillModel
	models::M 
	m::O
end

Flux.@functor(Ensemble)


function (me::Ensemble)(ds...)
	o = output(me.models[1](ds...))
	for j in 2:length(me.models)
		o = o + output(me.models[j](ds...).data)
	end
	ArrayNode(o ./ length(me.models))
end


Mask(ds::AbstractNode, me::Ensemble, statsfun, clustering) = Mask(ds, me.models[1], statsfun, clustering)