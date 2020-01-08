```@meta
EditURL = "<unknown>/README.jl"
```

ExplainMill.jl

This library provides an explanation of hierarchical multi-instance learning models using the method of Shapley values. This explanation method works by randomly perturbing "features" of the sample and observing the output of the classifier. The rationale behind the method is that if the feature is non-informative than the output of the classifier should be non-sensitive to its perturbation (and vice-versa). To support hierarchical models, we define several types of "daf" explainers.
* `BagDaf` explains `Mill.BagNode` by removing instances
* `TreeDaf` is just a container and does not explain anything
* `ArrayDaf` explains dense matrices (`Mill.ArrayNode{Matrix}`) by removing features of all samples (instances). Explaining individual items in dense matrices can be easily added, but at the moment author (Pevnak) is more interested in the influence of a particular feature on the sample. Features are modified by setting them to zero. This method assumes that features are centred, which is considering the state of JsonGrinder unlikely. A better method would be to provide a set samples such that values of modified features are replaced with a random sample from the population.
* `SparseArrayDaf` explains sparse matrix by zeroing individual items of the random matrix.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

