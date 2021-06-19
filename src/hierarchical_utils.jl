import HierarchicalUtils: NodeType, children, InnerNode, LeafNode, printtree, noderepr

# for schema structures
# TODO finish this
NodeType(::Type{T}) where T <: AbstractListMask = LeafNode()
NodeType(::Type{T}) where T <: EmptyMask = LeafNode()
NodeType(::Type{T}) where T <: BagMask = InnerNode()
NodeType(::Type{T}) where T <: ProductMask = InnerNode()

NodeType(::Type{T}) where T <: Absent = LeafNode()

noderepr(n::T) where {T <: AbstractStructureMask} = "$(T.name)"
noderepr(n::EmptyMask) = "skipped"
noderepr(::Mask{Nothing, D}) where {D} = "Simple Mask";
noderepr(::Mask{Vector{Int}, D}) where {D} = "Mask with clustering";

children(n::BagMask) = (n.child,)
children(n::ProductMask) = n.childs
