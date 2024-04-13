function confidence(model, ds)
    o = mapslices(x -> maximum(x), model(ds), dims=1)
    numobs(ds) == 1 ? o[1] : o
end

function confidencegap(o, i::Int)
    mx = mapslices(o, dims=1) do x
        ii = sortperm(x, rev=true)
        ii[1] == i ? x[ii[2]] : x[ii[1]]
    end
    o[i:i, :] .- mx
end

function confidencegap(o, classes::Vector{Int})
    @assert size(o, 2) == length(classes)
    map(1:length(classes)) do i
        x = @view o[:, i]
        ii = sortperm(x, rev=true)
        j = (ii[1] == classes[i]) ? ii[2] : ii[1]
        x[classes[i]] - x[j]
    end
end

confidencegap(model, ds, classes) = confidencegap(model(ds), classes)

"""
    logitconfgap(model, ds, classes)

    confidence gap assuming the output of the model 
    are log of probabilities (i.e. no softmax)
"""
logitconfgap(model, ds, classes) = confidencegap(softmax(model(ds)), classes)
logitconfgap(o::Matrix, classes) = confidencegap(softmax(o), classes)


Flux.onecold(model::Mill.AbstractMillModel, ds::Mill.AbstractMillNode) = Flux.onecold(softmax(model(ds)))
