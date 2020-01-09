module ExplainMill
using Mill, Duff, SparseArrays, StatsBase
using Mill: paddedprint, COLORS
import Mill: dsprint

abstract type AbstractDaf end;
Base.show(io::IO, ::MIME"text/plain", n::AbstractDaf) = dsprint(io, n)

include("densearray.jl")
include("sparsearray.jl")
include("bags.jl")
include("product.jl")

Duff.update!(daf, mask::Nothing, v::Number, valid_indexes = nothing) = nothing


function explain(ds, model, nsamples=10000)
	daf = Duff.Daf(ds);
	for i in 1:nsamples
		dss, mask = sample(daf, ds)
		v = model(dss).data[1]
		Duff.update!(daf, mask, v)
	end
	return(daf)
end

export explain, prune
end # module
