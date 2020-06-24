import HierarchicalUtils: NodeType, children, InnerNode, LeafNode, printtree, noderepr

# for schema structures
NodeType(::Type{T}) where T <: AbstractListMask = LeafNode()
NodeType(::Type{T}) where T <: EmptyMask = LeafNode()
NodeType(::Type{T}) where T <: BagMask = InnerNode()
NodeType(::Type{T}) where T <: TreeMask = InnerNode()

noderepr(n::AbstractExplainMask) = "$(Base.typename(typeof(n)))"
noderepr(n::EmptyMask) = "skipped"
string(Base.typename(BagMask))

children(n::BagMask) = (n.child,)
children(n::TreeMask) = (; n.childs...)
