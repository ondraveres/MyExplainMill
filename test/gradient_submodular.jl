using Mill
using JsonGrinder
using ExplainMill
using Flux
using ExplainMill: create_mask_structure

a = ArrayNode(reshape(collect(1:10), 2, 5))
b = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 123, 256))
c = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
ds = BagNode(ProductNode((;a, b, c)), AlignedBags([1:2,3:3,4:5]))
model = reflectinmodel(ds, d -> Dense(d, 4), d -> SegmentedMeanMax(d))


create_mask = d -> ExplainMill.Mask(GradientMask(ones(Float32, d)), fill(true, d), fill(0, d), nothing, nothing)
mk = create_mask_structure(ds, model, create_mask, ExplainMill._nocluster)

mk = BagMask(ProductMask(


	))