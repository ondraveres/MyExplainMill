using ExplainMill, Test
using ExplainMill: jsondiff

d1 = Dict(:a => [1, 2], :b => ["a", "c"], :c => Dict(:a => "hello"))
d2 = Dict(:a => [1, 2, 4], :b => ["b", "c"])
jsondiff(d1, d2)
jsondiff(d2, d1)

@test ExplainMill.nleaves(d1) == 5
@test ExplainMill.nleaves(d2) == 5

d1 = Dict("a" => [1, 2], "b" => ["a", "c"], "c" => Dict("a" => "hello"))
d2 = Dict(:a => [1, 2, 4], :b => ["b", "c"])


explanation = Dict{Symbol,Any}(:lumo => -1.492, :inda => 0.0, :mutagenic => 1.0, :atoms => Any[])
concept = Dict{String,Any}("lumo" => -1.492, "inda" => 0.0, "mutagenic" => 1.0)
exMinusConcept = jsondiff(explanation, concept)
conceptMinusEx = jsondiff(concept, explanation)

@test exMinusConcept == Dict{Any,Any}()
@test conceptMinusEx == Dict{Any,Any}()



explanation2 = Dict{Symbol,Any}(:lumo => -1.492, :inda => 0.0, :mutagenic => 1.0, :atoms => Any["c", "d"])
concept2 = Dict{String,Any}("lumo" => -1.492, "inda" => 0.0, "mutagenic" => 1.0)
exMinusConcept2 = jsondiff(explanation2, concept2)
conceptMinusEx2 = jsondiff(concept2, explanation2)

@test exMinusConcept2 == Dict{Symbol,Vector{Any}}(:atoms => ["c", "d"])
@test conceptMinusEx2 == Dict{Any,Any}()