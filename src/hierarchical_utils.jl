import HierarchicalUtils: NodeType, childrenfields, children, InnerNode, SingletonNode, LeafNode, printtree, noderepr

# for schema structures
NodeType(::Type{T}) where T <: BagMask = SingletonNode()
NodeType(::Type{MatrixMask}) = LeafNode()
NodeType(::Type{T}) where T <: TreeMask = InnerNode()
NodeType(::Type{SparseArrayMask}) = LeafNode()
NodeType(::Type{NGramMatrixMask}) = LeafNode()

noderepr(n::BagMask) = "BagMask"
noderepr(n::MatrixMask) = "MatrixMask"
noderepr(n::TreeMask) = "TreeMask"
noderepr(n::SparseArrayMask) = "SparseArrayMask"
noderepr(n::NGramMatrixMask) = "NGramMatrixMask"

childrenfields(::Type{T}) where T <: BagMask = (:child,)
childrenfields(::Type{TreeMask}) = (:childs,)

children(n::BagMask) = (n.child,)
children(n::TreeMask) = (; n.childs...)
