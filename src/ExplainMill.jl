module ExplainMill
using Mill, Duff, SparseArrays, StatsBase, CatViews
using Mill: paddedprint, COLORS
import Mill: dsprint

abstract type AbstractDaf end;
Base.show(io::IO, ::MIME"text/plain", n::AbstractDaf) = dsprint(io, n)

include("densearray.jl")
include("sparsearray.jl")
include("NGramMatrix.jl")
include("skip.jl")
include("bags.jl")
include("product.jl")
include("explain.jl")

Duff.update!(daf, mask::Nothing, v::Number, valid_columns = nothing) = nothing

export explain, dafstats

end # module
