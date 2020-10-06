using Test
using ExplainMill: findnonempty
@test findnonempty(ArrayNode(randn(2,0))) == nothing
@test findnonempty(ArrayNode(randn(2,1))) == [(@lens _.data)]
@test findnonempty(BagNode(ArrayNode(randn(2,0)), [0:-1])) == nothing
@test findnonempty(BagNode(ArrayNode(randn(2,1)), [1:1])) == [(@lens _.data.data)]
@test findnonempty(ProductNode((a = BagNode(ArrayNode(randn(2,0)), [0:-1]), ))) == nothing
# findnonempty(ProductNode((a = BagNode(ArrayNode(randn(2,1)), [1:1]), ))) .== [(@lens _.data.a.data.data)]
# findnonempty(ProductNode((a = BagNode(ArrayNode(randn(2,1)), [1:1]), b = ArrayNode(randn(2,1))))) == ([(@lens _.data.a.data)], [(@lens _.data.b)])
# findnonempty(ProductNode((a = BagNode(ArrayNode(randn(2,0)), [0:-1]), b = ArrayNode(randn(2,1))))) == ([(@lens _.data.b)],)
