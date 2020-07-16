using ExplainMill
# TODO export these?
using ExplainMill: prune, Mask, invalidate!, mapmask, participate, mask, prunemask
using ExplainMill: MatrixMask, SparseArrayMask, CategoricalMask, ProductMask, BagMask
using Test
using Mill
using Flux
using Duff
using SparseArrays

include("maskstructure.jl")
include("flatmasks.jl")
include("gnn.jl")

# TODO worth it?
include("explain.jl")
include("sigmoid.jl")
