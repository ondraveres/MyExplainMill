function StatsBase.predict(model::Mill.AbstractMillModel, ds::Mill.AbstractNode, ikeyvalmap)
	o = mapslices(x -> ikeyvalmap[argmax(x)], model(ds).data, dims = 1)
    nobs(ds) == 1 ? o[1] : o
end

function StatsBase.predict(model::Mill.AbstractMillModel, ds::Mill.AbstractNode)
	o = mapslices(x -> argmax(x), model(ds).data, dims = 1)
    nobs(ds) == 1 ? o[1] : o
end

function confidence(model, ds)
	o = mapslices(x -> maximum(x), model(ds).data, dims = 1)
    nobs(ds) == 1 ? o[1] : o
end

function confidencegap(o, correct_class::Int)
    i = correct_class
    mx = mapslices(o, dims = 1) do x 
        ii = sortperm(x, rev = true)
        ii[1] == i ? x[ii[2]] : x[ii[1]]
    end
    o[i:i,:] .- mx
end
confidencegap(model, ds, correct_class::Int) = confidencegap(model(ds).data, correct_class)

function confidencegap1(model, ds, correct_class::Int)
	@assert nobs(ds) == 1 
	confidencegap(model, ds, correct_class)[1]
end
