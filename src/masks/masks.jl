abstract type AbstractStructureMask end;
abstract type AbstractVectorMask end;
abstract type AbstractListMask <: AbstractStructureMask end;
abstract type AbstractNoMask <: AbstractStructureMask end;

"""
	invalidate!(m::AbstractStructureMask, obs::Vector{Int})

	set participation to `false` observations `obs` and their childs
"""
invalidate!(m::AbstractStructureMask) = invalidate!(m, Vector{Int}())

"""
	support_participation(m::AbstractVectorMask)

	`true` if the mask `m` supports tracking of participation, 
	which is required for level-by-level explanations
"""
support_participation(m::AbstractVectorMask) = false

"""
	updateparticipation!(mk)

	update the information if an item of masks has effect on 
	the output `participation` according to current settings 
	of masks
"""
function updateparticipation!(mk)
	foreach_mask((m, level) -> participate(m) .= true, mk)
	invalidate!(mk)
end

"""
	create_mask_structure(f, ds::Mill.AbstractMillNode)
	create_mask_structure(ds::Mill.AbstractMillNode, f)

	construct structural masks for a sample `ds` while initating 
	`<:AbstractVectorMask` using function `f`.

Example:

```julia
julia> create_mask_structure(ArrayNode(randn(3,4)), d -> SimpleMask(fill(true, d)))
typename(MatrixMask)
```
"""
create_mask_structure(f, ds::Mill.AbstractMillNode) = create_mask_structure(ds, f)

mapmask(f, m::AbstractStructureMask) = mapmask(f, m, 1, IdDict{Any,Any}())
foreach_mask(f, m::AbstractStructureMask) = foreach_mask(f, m, 1, IdDict{Any,Nothing}())

include("participation.jl")
include("simplemask.jl")
include("mask.jl")
include("parentstructure.jl")
include("flatview.jl")

include("nodemasks/observationmask.jl")
include("nodemasks/densearray.jl")
include("nodemasks/sparsearray.jl")
include("nodemasks/categoricalarray.jl")
include("nodemasks/ngrammatrix.jl")
include("nodemasks/skip.jl")
include("nodemasks/bags.jl")
include("nodemasks/product.jl")
include("nodemasks/lazymask.jl")


Base.length(m::AbstractStructureMask) = length(m.mask)
Base.getindex(m::AbstractStructureMask, i) = m.mask[i]
Base.setindex!(m::AbstractStructureMask, v, i) = m.mask[i] = v