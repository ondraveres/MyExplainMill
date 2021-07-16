# How and what to test

* Structural Masks should be tested as is written in `test/structuralmasks.jl`
* Compatibility of simplemask with an api used in pruning is in `test/flatmask.jl`. It should be sufficient to add a new constructor to the loop.