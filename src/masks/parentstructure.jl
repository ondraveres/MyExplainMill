const ParentStructure = Vector{Pair{AbstractStructureMask, Int}}
"""
	collectmasks(mk::AbstractStructureMask, level = 1)

	recursively traverse the hierarchical mask structure and 
	collect mask that have explainable content together with their level.

example:
```julia
julia> an = ArrayNode(randn(4,5))
ArrayNode(4, 5)

julia> ds = BagNode(an, AlignedBags([1:2,3:3,0:-1,4:5]))
BagNode with 4 bag(s)
  └── ArrayNode(4, 5)

julia> mk = create_mask_structure(ds, d -> SimpleMask(fill(true, d)))
typename(BagMask)
  └── typename(MatrixMask)

julia> collectmasks(mk)
2-element Vector{Any}:
 SimpleMask{Bool}(Bool[1, 1, 1, 1, 1]) => 1
    SimpleMask{Bool}(Bool[1, 1, 1, 1]) => 2

```
"""
function collectmasks(mk, level = 1)
	collected_masks = []
	mapmask((mk, depth) -> push!(collected_masks, mk => depth), mk, level)
	collected_masks
end