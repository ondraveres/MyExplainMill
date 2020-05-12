using ExplainMill, Test
using ExplainMill: jsondiff

d1 = Dict(:a => [1,2], :b => ["a","c"], :c => Dict(:a => "hello"))
d2 = Dict(:a => [1,2,4], :b => ["b","c"])
jsondiff(d1, d2)
jsondiff(d2, d1)

@test ExplainMill.nleaves(d1) == 5
@test ExplainMill.nleaves(d2) == 5

d1 = Dict("a" => [1,2], "b" => ["a","c"], "c" => Dict("a" => "hello"))
d2 = Dict(:a => [1,2,4], :b => ["b","c"])
