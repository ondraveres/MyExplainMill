const ItemMap = Array{NamedTuple{(:itemid, :maskid, :innerid),Tuple{Int64,Int64,Int64}},1}

"""
	A structure presenting masks in the PruiningMask structure
	as a flat array of coefficients, ideally sorted according to 
	chosen value. Hence, the user does not have to be aware of the structure
	tree structure, which is abstracted as flat array. The structure supports
	indexing through `setindex` and `getindex` and it also returns the parent 
	of an item in `parent`, which points to an item which can influence the given 
	point.
"""
struct FlatView{V}
	masks::V
	itemmap::ItemMap
	starts::Vector{Int}
end

Parents = Array{Pair{k,Int64} where k,1}

function FlatView(mk::AbstractStructureMask)
	collected_masks = []
	foreach_mask((mk, depth) -> push!(collected_masks, mk), mk, 1)
	FlatView(collected_masks)
end

function FlatView(masks::Vector)
	itemno = 0
	itemmap = map(enumerate(masks)) do (i,m)
		map(1:length(m)) do j
			itemno += 1
			(itemid = itemno, maskid = i, innerid = j)
		end
	end
	itemmap = reduce(vcat, itemmap)
	starts = vcat(0, accumulate(+,map(m -> length(m), masks))[1:end-1])
	FlatView(tuple(masks...), itemmap, starts)
end

function Base.setindex!(m::FlatView, v, i::Int) 
	_, k, l = m.itemmap[i]
	m.masks[k][l] = v
end


"""
	copyto!(fv::FlatView, bc)

	sets masks in `fv` to bc ignoring all dependencies
"""
function Base.copyto!(fv::FlatView, bc)
	for (i, m) in enumerate(fv.masks)
		offset = fv.starts[i]
		for j in 1:length(m)
			m[j] = bc[j + offset]
		end
	end
end

# # this is a syntactic sugar such that I can write fv .= x
# function Base.materialize!(fv::FlatView, a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1},Nothing,typeof(identity),Tuple{Array{Bool,1}}} )
# 	copyto!(fv, a.args[1])
# end

function Base.getindex(m::FlatView, i::Int) 
	_, k, l = m.itemmap[i]
	m.masks[k][l]
end

function Base.getindex(m::FlatView, ii::Vector{Int}) 
	map(i -> m[i], ii)
end


Base.fill!(fv::FlatView, v) = foreach(i -> fv[i] = v, 1:length(fv))
Base.ndims(::Type{<:FlatView}) = 1
Base.length(m::FlatView) = length(m.itemmap)
Base.size(fv::FlatView) = (length(fv),)

useditems(m::FlatView) = findall(usedmask(m))
usedmask(m::FlatView) = map(i -> m[i], 1:length(m)) #this is really ineffective but that is life
participate(m::FlatView) = reduce(vcat, map(participate, m.masks))
support_participation(m::FlatView) = all(support_participation(i) for i in m.masks)
heuristic(m::FlatView) = reduce(vcat, map(heuristic, m.masks))
copy2vec(m::FlatView) = reduce(vcat, map(prunemask, m.masks))
