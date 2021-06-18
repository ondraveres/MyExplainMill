# Mask structure

Explanation is all about significance and masking of terms. What does it mean? We see an explanation as a solution of the following problem:

``min ||m||₀``

subject to 
``f(x .* m)[y] >= τ	 ``

i.e. we are looking for an `m` of minimum cardinality such that we are classifying sample `x` on a subset of `m` with a sufficient confidence. Since `m` is technically a binary vector (`true`, `false`), the problem is NP complete and therefore there is no single approach to deal with this. List below is a ad-hoc list of approaches and requirements we have on the implementation of mask `m`. 

* Plethora of works deal with the above by assigning each item (also called term to resemble the logic interpretation) a significance (a heuristic value) and then use this significance as a heuristic guiding the removal of terms.

* In `gradient_submodular` the significance is updated during the pruning, threfore the significance should give better information about the local optimization landscape of the above, but it is a very different paradigm from the above.

* If the input sample `x` is structured (e.g. like Mill), mask `m` should be structured as well. But for cleanliness of api, parts of it (for example a level in the tree) should be representable as a flat vector, therefore search algorithms can work over an abstraction (this is taken care of by a `FlatView`).

* If multiple samples are explained simultaneously, it is advantageous to cluster similar terms it `m` representing similar (same) objects together. This has the advantage that the resulting explanation is more coherent across all samples.

* Some heuristic methods needs to take gradient with respect to `m`. 

* Some methods, for example GNNExplainer considers represents the mask to be unconstrained and maps them to `[0,1]` interval using `σ`. Contrary, Gradient heuristics considers mask as `[0,1]` and calculate gradient as is.

* If hierarchical structure is taken into the account during explanation, it makes a lot of sense to be able to identify non-participating items being effectively blocked by their parents. This is especially important to children of  `BagNodes`. *Should this information be stored inside the Mask?* This information is also used in Bazhaff values.

* To calculate Banzhaf values for a batch of samples, each item in a mask has to be linked to the item of the input.

### How we deal with it at currently?

Poorly of course. The structural (hierarchical) aspect of masks is kept in special mask reflectining nodes in `Mill.jl` hierarchies. Therefore there is `ProductMask`, `BagMask`, various types of `ArrayMasks` for different types of arrays (Sparse, Categorical, Dense, NGram...), and `GraphGrinder.jl` defines mask for Graphs. Furthermore, there is `EmptyMask` used to signal either there is nothing to explain or the node is excluded from the search. 


Each of the above holds one (or more) `Mask` objects
```julia
struct Mask{I<:Union{Nothing, Vector{Int}}, D}
	mask::Vector{Bool}
	participate::Vector{Bool}
	outputid::Vector{Int}
	stats::D
	cluster_membership::I
end
```

containing information about:
* which items of the node in the structure is active (a boolean vector `mask`);
* is the item blocked by its parents? (a boolen vector `participate`);
* to which sample in a batch the item belongs to (`outputid`), exclusively used only in the computation of Shapley / Banzhaf values of a batch;
* importance values of each item (in `stats`);
* mapping from clusters to items if they are clustered.

Problems ?
* Methods requiring gradients (GNNExplainer, Grad methods) uses two masks, as in `stats` they store the differentiable mask and in `mask` they store the pruning mask. Pruning mask is accessed by `prunemask` used in subsetting the sample `ds` as `ds[m]`, while differentiable mask is accessed by `diffmask` used in differential evaluation of a sample by a `model` as `model(ds, m)`.
* 
* `diffmask` has to output the correctly masked item, therefore for GNNExplainer the `diffmask(m) = σ(m.stats)` but for `GradExaplainer` it should be `diffmask(m) = m.stats` (this problem is currently caused by insufficient dispatch). Moreover, `model(ds, m) ≠ model(ds[m])`, which causes discrepancy between the view on the sample of gradient based beuristics and that of the sub-sampling based heuristics. 
* Is participation really needed? Can we get away with it? It is used mainly in Shapley, where it makes a difference between Banzhaf and Shapley. It is also used in pruning, where it helps to identify parts of masks, which does not have to be optimized over?
* `outputid` is used only by Shapley values.


### A working idea
I would like to abstract the structure (or its part) such that it will behave as a vector. That means that I can access it and modify it using `setindex!`, `getindex`, `length`, and `empty`. Moreover, for a given set of masks, I would like to get a heuristic values. At the same time, whole sample should be indexable by a a structural mask.

So the current idea would be to have:
* structural Mask elements, that are specialized for corresponding Node Elements. So `BagNode` has a `BagMask`, `ProductNode` has a `ProductMask`, `ArrayNode` has `ArrayMask`, etc.
* Each structural mask will contain a `SomeMask` which will behave like a vector. It will implement
	- `prunemask(m)` used for subsetting of a sample `ds[prunemask(m)]`
	- `diffmask(m)` used differentiable simulation of subsetting `model(ds, diffmask(m))`
	- `setindex!(m, i, ::Bool)` for pruning
	- `getindex(m, i)` to get current values
	- `length(m)` to get number of items
	- `isempty(m)` to get number of items
	- `heuristic(m)` which would provide heuristic at a given state, but it can be for example state-independent (e.g. Banzhaf, etc...)

* The heuristic can specialize on `SomeMask`, which can be special for a  combination of Heuristic and `Mask`. This would allow to solve issue of GnnExplainer requiring a different transformation then GradExplainer and / or sub-modular grad method. 

* The `ClusteredMask` can wrap `SomeMask` (or vice versa).

#### Discussion

It is not clear to me, if the above is not over-engineered and / or sufficiently general, as there are many open questions. 
* Can get rid of `participation` everywhere except the Shapley / Bazhaffs.
* Can we support multi banzhaf values?
* 