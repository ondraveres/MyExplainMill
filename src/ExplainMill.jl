module ExplainMill
using Mill, Duff, SparseArrays, StatsBase, CatViews
using Mill: paddedprint, COLORS
import Mill: dsprint
using TimerOutputs

const to = TimerOutput();

abstract type AbstractExplainMask end;
abstract type AbstractListMask <: AbstractExplainMask end;


struct Mask
	mask::Array{Bool,1}
	participate::Array{Bool,1}
end

Mask(d::Int) = Mask(fill(true, d), fill(true, d))

participate(m::Mask) = m.participate
participate(m::AbstractListMask) = participate(m.mask)
mask(m::Mask) = m.mask
mask(m::AbstractListMask) = mask(m.mask)

function mapmask(f, m::AbstractListMask)
	(mask = f(m.mask),)
end

invalidate!(m::AbstractExplainMask) = invalidate!(m, Vector{Int}())

Base.show(io::IO, ::MIME"text/plain", n::AbstractExplainMask) = dsprint(io, n)

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
