# ExplainMill.jl

This library provides an explanation of hierarchical multi-instance learning models using the method of Shapley values. This explanation method works by randomly perturbing "features" of the sample and observing the output of the classifier. The rationale behind the method is that if the feature is non-informative than the output of the classifier should be non-sensitive to its perturbation (and vice-versa). To support hierarchical models, we define several types of "daf" explainers.
* `BagDaf` explains `Mill.BagNode` by removing instances
* `TreeDaf` is just a container and does not explain anything
* `ArrayDaf` explains dense matrices (`Mill.ArrayNode{Matrix}`) by removing features of all samples (instances). Explaining individual items in dense matrices can be easily added, but at the moment author (Pevnak) is more interested in the influence of a particular feature on the sample. Features are modified by setting them to zero. This method assumes that features are centred, which is considering the state of JsonGrinder unlikely. A better method would be to provide a set samples such that values of modified features are replaced with a random sample from the population.
* `SparseArrayDaf` explains sparse matrix by zeroing individual items of the random matrix.

Let's demonstrate the usage. Assume that we have sample `ds` and model `model`, and we want to calculate daf statistics using 1000 random subsamples of `ds`.

```@example README
using Mill, ExplainMill, SparseArrays, Flux

an = ArrayNode(randn(2,5))
cn = ArrayNode(sprand(2,5, 0.5))
bn = BagNode(TreeNode((a = an, c = cn)), AlignedBags([1:2,3:5]))
tn = TreeNode((a = an[1:2], b = bn))
ds = BagNode(tn, AlignedBags([1:2]))
model = reflectinmodel(ds, d -> Dense(d,2), d -> SegmentedMean(d), b = Dict("" => d -> Dense(d,1)))

daf = ExplainMill.explain(ds, model, 1000)
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

