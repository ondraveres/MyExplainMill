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
* `outputid` is used only by Shapley values.


### A working idea
I would like to abstract the structure (or its part) such that it will behave as a vector. That means that I can access it and modify it using `setindex!`, `getindex`, `length`, and `empty`. Moreover, for a given set of masks, I would like to get a heuristic values. At the same time, whole sample should be indexable by a a structural mask.

So the current idea would be to have:
* structural Mask elements, that are specialized for corresponding Node Elements. So `BagNode` has a `BagMask`, `ProductNode` has a `ProductMask`, `ArrayNode` has `ArrayMask`, etc. Structural mask will have a following interface:
	- `Base.getindex(sample, mask)` returns a subset of a sample as specified by the mask. This uses `prunemask` to obtain pruning mask from the `<:AbstractVectorMask`
	- `invalidate!(mask, invalid_observations)` marks `invalid_observation` and their descendants as not participating in the explanation / pruning search (see belows on problematics of `participation`)
	- `(m::Mill.AbstractModel)(sample, mask)` output of a model `m` on a `sample[mask]`. This implementation should be differentiable and the `mask` can be between zero and one. Note that it should be the case that `m(ds, mask) == m(ds[mask])` if items of `mask` are `{true,false}`. 
	- `foreach_mask(f, mask)` applies function `f` on `<:AbstractVectorMask` of a given node and its child. The argument of `f` is a tuple `(<:AbstractVectorMask, depth::Int)`, where `depth` is the distance of a current `mask::AbstractVectorMask` from the top node. It is responsibility of the implementation to increase the depth accordingly. This allows to apply `f` on on masks at some level. The overloading allows not to count some `AbstractStructureMask` if they do not play active role (e.g. `ProductMask`) and to impose structure if some `<:AbstractStructureMask` contain more then one `<:AbstractVectorMask`. Contrary, `BagMask` should increase the level, since it contains a mask which directly influences masks of its child.
	- `mapmask(f, mask)` applies function `f` on `<:AbstractVectorMask` of a given node and its child and return modified structural mask `AbstractStructureMask`. For example for `Matrix`, it would look like this 
	```
	function mapmask(f, m::MatrixMask, level = 1)
		MatrixMask(f(m.mask, level))
	end
	```
	This function is more of a convenience and nice to have things, as it allows to separate mask used to calculate heuristic values from mask used in the pruning process, where it might be sufficient just to provide heuristic values. If differentiable mask undergoes some complicated transformation, e.g. `σ`, it is not clear, what to use for a heuristic values.
	- `partialeval` 	identify subset of `model`, sample `ds`, and structural mask `mk` that are sensitive to `masks` and evaluate and replace the rest of the model, sample, and masks by `identity`, `ArrayNode`, and `EmptyMask`. Partial evaluation is useful when we are explaining only subset of full mask (e.g. level-by-level) 
	explanation.


* Each structural mask will contain a `SomeMask<:AbstractVectorMask` which will behave like a vector. It will implement
	- `prunemask(m)` used for subsetting of a sample `ds[prunemask(m)]`
	- `diffmask(m)` used differentiable simulation of subsetting `model(ds, diffmask(m))`
	- `setindex!(m, i, ::Bool)` for pruning
	- `getindex(m, i)` to get current values
	- `length(m)` to get number of items
	- `isempty(m)` to get number of items
	- `simplemask` would be the `AbstractVectorMask` stripped of all decorators. In simplest case, it can be just `simplemask(m) = m`. It should be the parameter that can `Zygote` accepts and we can take gradient over.
	- `heuristic(m)` which would provide heuristic at a given state, but it can be for example state-independent (e.g. Banzhaf, etc...)

* The heuristic can specialize on `SomeMask`, which can be special for a  combination of Heuristic and `Mask`. This would allow to solve issue of GnnExplainer requiring a different transformation then GradExplainer and / or sub-modular grad method. 

* The `ClusteredMask` can wrap `SomeMask` (or vice versa).

#### Discussion

It is not clear to me, if the above is not over-engineered and / or sufficiently general, as there are many open questions. 
* Can we support multi banzhaf values?
* I do not know, if the FlatView should contain `StructureMask` or `Mask`. The latter has the advantage that it has a clearly defined 

#### What needs to be tested and verified for structured masks
The goal is to put as much test related to a single `Type` in one block, such that when someone will be introducing new type, he can see all tests in a single block of code.

* Test that matrix is correctly created and full matrix does not change the element
```julia
	an = ArrayNode(randn(4,5))
	mk = create_mask_structure(an, d -> SimpleMask(fill(true, d)))
	@test mk isa MatrixMask
	@test an[mk] == an
```		
* Verify pruning has an effect

```julia
	prunemask(mk.mask)[[1,3]] .= false
	@test prunemask(mk.mask) == [false, true, false, true]

	# testing that subsetting works as it should
	@test an[mk].data ≈ an.data .* [false, true, false, true]	
```
* Test that multiplication and masking has the same effect (more on that below). Notice that weigths of the model is converted to `Float64` for the sake of analysis.
```julia
	# testing that multiplication is equicalent to subsetting
	model = reflectinmodel(an, d -> Dense(d, 10))
	@test f64(model(an[mk]).data ≈ model(an, mk).data)
```
* Test that we can calculate the gradient with respect to diff mask
```julia
	# We should test that we can calculate gradent
	gs = gradient(() -> sum(model(an, mk).data),  Flux.Params([mk.mask.x]))
	# ∇mk = gs[mk.mask.x]
	@test sum(abs.(gs[mk.mask.x])) > 0
```

* Verify that calculation of the gradient for real mask is correct 
```julia
	mk = create_mask_structure(an, d -> SimpleMask(rand(d)))
	ps = Flux.Params([mk.mask.x])
	testmaskgrad(() -> sum(model(an, mk).data),  ps)
	@test all(abs.(gs[mk.mask.x]) .> 0)
```

#### Differentiable mask is not trivial --- simple arrays
Recall the need for differentiable mask is that some heuristic methods likes to calculate gradient with respect to the mask. Therefore we need 
to calculate gradient with respect to the mask (correctly), which might not be tricky.

As an example, consider masking categorical variable. The *absence* is commonly (and done in Mill) represented as a special category.  For example we have a categorical matrix with `x = [1,2,3,1,2]` with categories `[1,2,3]` and [4] reserved for missing. Such `x` can be represented as a sparse Matrix
```julia
julia> x = sparse(ds.data)
4×5 SparseMatrixCSC{Bool, Int64} with 5 stored entries:
 1  ⋅  ⋅  1  ⋅
 ⋅  1  ⋅  ⋅  1
 ⋅  ⋅  1  ⋅  ⋅
```
If we apply mask `m = [false,true,false,true,true]`, the masked `x` (denoted as `y` will look like 
```julia
julia> prunemask(mk.mask)[[1,3]] .= false;

julia> y = sparse(ds[mk].data)
4×5 SparseMatrixCSC{Bool, Int64} with 5 stored entries:
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  1  ⋅  ⋅  1
 ⋅  ⋅  ⋅  ⋅  ⋅
 1  ⋅  1  ⋅  ⋅
```

Therefore to make the output of function `f` differentiable with respect to the mask `m` indicating if the item is present or absent,
```julia
f(@. m * x + (1 - m) .* y′)
```
where `y′` is a canonical representation of a missing vector.This is very different from the usual "setting that to zero". Crucially, the gradient with respect to `m` needs to depend on a difference between `x` and `y`. Needless to say, if `y′` would be set to zero, it would be simple as the second term disappears to  `f(@. m * x)′` and everything boils down to a fundamental question if `zero` is the right value for missing feature.

#### Differentiable mask is not trivial --- aggregation functions

How shall the mask works on bags? Let's assume the bag consists of n instances `(x_1,...,x_n).` and assume a mask `(m_1,...,m_n).` If items of mask are from `{0,1}`, then `0` indicates that item is absent and `1` indicates it is present. If `A` is an aggregation function, then we want that the output on masked `(x_1,...,x_n)` to be equal to output on items where mask is equal to one `A ({x_i}_{m_i == 1}).` Imagine now that items of mask can have any values from `[0,1].` We would like to have continuitity, i.e. ` A((x_1 * m_1,...,x_n * m_n)) ⟶ A(∅)` as `\|m\| ⟶ 0`. Similarly, we want `A((x_1 * m_1,...,x_{j-1} * m_{j-1},x_j * m_{j},x_j+1 * m_{j+1}, ..., x_n * m_n)) ⟶ A((x_1 * m_1,...,x_{j-1} * m_{j-1},x_j+1 * m_{j+1}, ..., x_n * m_n))` as `|m_j| ⟶ 0. ` This implies that 
* for `A` being ` ∑` or `mean` `A(∅) = 0`.
* for `A` being `maximum` `A(∅) = minimum({x})` where the minimum goes over the all items in the training set.

### Is participation needed? 

The `participation` allows to track, if a part of a mask has an effect of the output of a classifier on a given sample. For example imagine that we have a following sample 
```
ds = BagNode(
	ArrayNode(
		NGramMatrix(["a","b","c","d","e"])
		)
	[1:5]
	)
```

`BagMask` is masking individual observations and so does `NGramMatrix`. This means that if `BagMask` has mask `[true,false,true,false,true]`, then explanation of `NGramMatrix` can consider only `["a","c","e"]` instead of all items, because `["b","d"]` are removed by the `BagMask` of above. The `participation(mk::NGramMatrixMask)` should therefore return `[true,false,true,false,true]`.

An interesting question is how to implement participation. So far, I have considered three options.
* Each structural mask will have `participation` field, which can be updated. The idea behind this is that to update `participation`, the structure and type of data is important. This would make it independent to how `<:AbstractVectorMask` is implemented, but it would not fly, because the `participation` might be influenced by *Clustering* of masks.
* The other approach is to make a `participation` field a mandatory part of `<:AbstractVectorMask`, but that seems to me unnecessary burden, since it `<:AbstractVectorMask` should not care about it. At least, at the moment, I do not see a reason why it should.
* The third approach, which I kind of like it to implement `participation` as a decorator of `<:AbstractVectorMask`. It would reexport api of the `<:AbstractVectorMask`, therefore it would be invisible to it and also it would allow to override it for the clustering.  
* We should have a trait to know, if the participation is supported and behave accordingly.