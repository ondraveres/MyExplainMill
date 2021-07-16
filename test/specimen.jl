function specimen_sample()
	an = ArrayNode(reshape(collect(1:10), 2, 5))
	on = ArrayNode(Flux.onehotbatch([1, 2, 3, 1, 2], 1:4))
	cn = ArrayNode(sparse(Float32.([1 0 3 0 5; 0 2 0 4 0])))
	sn = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 256, 2053))
	ds = BagNode(BagNode(ProductNode((;an, on, cn, sn)), AlignedBags([1:2,3:3,4:5])), AlignedBags([1:3]))
end
