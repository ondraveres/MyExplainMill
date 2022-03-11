## Supported heuristics
* Non-informative heuristics Constant `ConstExplainer` and stochastic `StochasticExplainer` are implemented mainly as a reference to see the importance of heuristic and that of pruning step
* `GnnExplainer` is a heuristic based on variational approximation o a discrete mask optimizing trade-off between confidence in explanation and sparsity of the mask 
* `DafExplainer / ShapExplainer / BanzExplainer` are variants (and synonyms) to Shapley values also defined in the paper Dapendency Aware Feature selection.
* `GradExplainer` defines importance of a feature as an absolute value of the gradient of the output with respect to the mask. 

## Api
The api offers high-level function
`mask = stats(e, ds, model, classes = onecold(model, ds), clustering = _nocluster`
which uses an appropriate function for a given explanation method and if desired determines
explained class. `mask` is the StructureMask for a given sample ds.

Alternatively, you can override a function used to calculate the heuristic using `statsf(e, ds, model, f, clustering)`. In this case it is expected that your function is compatible with the explainer e. 
* For `StochasticExplainer` and `ConstExplainer` `f` does not matter at all.
* `BanzExplainer` and its variants and `GradExplainer` has default `f` equal to the "confidence" (output after sofmax) on the class that is being explainer
* `GnnExplainer` has `f` equal to logitcrossentropy


### Unit testing
Heuristic functions are tested in `heuristic.jl` in tests and their integration with 
