import HierarchicalUtils: NodeType, childrenfields, children, InnerNode, SingletonNode, LeafNode, printtree, noderepr

# for schema structures
NodeType(::Type{T}) where T <: AbstractListMask = LeafNode()
NodeType(::Type{T}) where T <: EmptyMask = LeafNode()
NodeType(::Type{T}) where T <: BagMask = SingletonNode()
NodeType(::Type{T}) where T <: TreeMask = InnerNode()

noderepr(n::AbstractExplainMask) = "$(Base.typename(typeof(n)))"
noderepr(n::EmptyMask) = "skipped"
string(Base.typename(BagMask))
childrenfields(::Type{T}) where T <: BagMask = (:child,)
childrenfields(::Type{TreeMask}) = (:childs,)

children(n::BagMask) = (n.child,)
children(n::TreeMask) = (; n.childs...)
