# Formalization of the Explanation and Rule creation


## Explanation

The explanation is understood as (a set of) minimal subset of the sample `x`, such that the output of a classifier `f(x)` on the class `y` is above thresholds `τ`. In a notation of a mathematical programming, the explanation solves the folowing problem

``\min \|m\|_0``

subject to 
``f(x .* m)[y] >= τ	 ``

where `m ∈ {0,1}^d`, where `d` is a dimension of `x`. `m` is therefore a mask indicating if the corresponding feature (or generally a subtree) is present in the sample or not.

The basic algorithm solving the above problem consists of the first (and optionally two more) phases:
1. **Addition** The algorithm starts with `m .= 0` and turns on a minimal number of items, such that the constraint `f(x .* m)[y] >= τ` is satisfied. I think that in the parlance of the optimization, we are looking for an interior point. The variants of this search includes: 
	* heuristics to priritize addition of important items first;
	* forward search, where in each iteration an item maximizing `f(x .* m)` is added.

2. **Removal** Once the initial solution is found, this step iteratively removes items from `m` until nothing can be removed while satisfying the constraint. This arrives to a local minima.

3. **Fine tuning** is a very expensive step inspired by Pudil's floating point scheme. The idea is to add (a few) items to be above the contraint and then to try a removal step.

### Analysis of the problem

The above algorithms are based on the assumptions of `f` that problem iff `m₁ ⊂ m₂` then `f(x .* m₁) ≤ f(x .* m₂).` `f` with this property is called submodular set function, but be aware that classifiers are satisfying this only 

## Rule creation for Clusty

The rule creation of clusty is a use-case, where given a sample `x`, one wants to create a subset of it (called rule), such that all samples that match this rule (the rule is their subset) will be of a same type (malware / cleanware) and of the same strain (if it applies). Thus, we want a rule which will have (i) low false positive rate, (ii) high true positive rate, (iii) and ideally high accuracy in identifying the given strain. Let `fₚ` and `fₙ` denotes the false positive rate and false negative rate.

An interesting questions is, what is a constraint and what is an optimization criterion. For example in the spirit of the above, one can 

``\min fₙ``

subject to 
``f(x .* m)[y] >= τ	 ``
``fₚ <= τₚ	 ``

where `fₙ` is minimized instead of the count of used items as there is a weak relationship, because adding items to a rule increases `fₙ` (there is chance to match lower number of items ). The advantage of this formulation is that only single criterion is minimized, which makes everything simple, but the optimization algorithm might choose a solution with worse `f(x .* m)[y]` if there is multiple solutions. 

An interesting property is that if `fₚ` and `fₙ` are estimated using a Naive Bayes approach, the objective and the second constraint are linear, but `f` is still highly nonlinear (but may-be closely subset submodular).

In other (not listed) formulation, some constraints are exchanged with the optimization term, but the overall structure and properties are the same.

### Some thoughts about solving of the above
* Due to the structure of the problem, a heuristic functions estimating importance of items might be replaced by the calculation of the gradient at a given point. This means that the solving of the above problems might be achieved using gradient descend problem **without any guarantees on optimality**.