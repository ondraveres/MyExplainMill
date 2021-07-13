abstract type AbstractStructureMask end;
abstract type AbstractVectorMask end;
abstract type AbstractListMask <: AbstractStructureMask end;
abstract type AbstractNoMask <: AbstractStructureMask end;

RealArray = Union{Vector{T}, Matrix{T}} where {T<:Real}

invalidate!(m::AbstractStructureMask) = invalidate!(m, Vector{Int}())

include("participation.jl")
include("simplemask.jl")
include("mask.jl")
include("parentstructure.jl")
include("flatview.jl")

include("nodemasks/densearray.jl")
include("nodemasks/sparsearray.jl")
include("nodemasks/categoricalarray.jl")
include("nodemasks/ngrammatrix.jl")
include("nodemasks/skip.jl")
include("nodemasks/bags.jl")
include("nodemasks/product.jl")
include("nodemasks/lazymask.jl")

function updateparticipation!(mk)
	foreach_mask((m, level) -> participate(m) .= true, mk)
	invalidate!(mk)
end

Base.length(m::AbstractStructureMask) = length(m.mask)
Base.getindex(m::AbstractStructureMask, i) = m.mask[i]
Base.setindex!(m::AbstractStructureMask, v, i) = m.mask[i] = v