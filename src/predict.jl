function predict(model, ds, ikeyvalmap)
	o = mapslices(x -> ikeyvalmap[argmax(x)], model(ds).data, dims = 1)
    nobs(ds) == 1 ? o[1] : o
end

function predict(model, ds)
	o = mapslices(x -> argmax(x), model(ds).data, dims = 1)
    nobs(ds) == 1 ? o[1] : o
end

function confidence(model, ds)
	o = mapslices(x -> maximum(x), model(ds).data, dims = 1)
    nobs(ds) == 1 ? o[1] : o
end