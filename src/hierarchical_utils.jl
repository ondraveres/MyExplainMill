import HierarchicalUtils: NodeType, children, InnerNode, LeafNode, printtree, nodeshow

# for schema structures
# TODO finish this
NodeType(::Type{T}) where T <: AbstractListMask = LeafNode()
NodeType(::Type{T}) where T <: EmptyMask = LeafNode()
NodeType(::Type{T}) where T <: BagMask = InnerNode()
NodeType(::Type{T}) where T <: ProductMask = InnerNode()

NodeType(::Type{T}) where T <: Absent = LeafNode()

nodeshow(io, (@nospecialize n::T)) where {T <: AbstractStructureMask} = print(io, "$(T.name)")
nodeshow(io, (@nospecialize n::EmptyMask)) = print(io, "skipped")
nodeshow(io, (@nospecialize n::Mask{Nothing, D})) where {D} = print(io, "Simple Mask";)
nodeshow(io, (@nospecialize n::Mask{Vector{Int}, D})) where {D} = print(io, "Mask with clustering";)

children(n::BagMask) = (n.child,)
children(n::ProductMask) = n.childs
