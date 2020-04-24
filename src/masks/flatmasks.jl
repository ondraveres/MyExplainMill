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


function FlatView(mask)
	masks = remove_useless_parents(parent_structure(mask))
	itemno = 0
	itemmap = map(enumerate(masks)) do (i,m)
		map(1:length(m.first.mask)) do j
			itemno += 1
			(itemid = itemno, maskid = i, innerid = j)
		end
	end
	itemmap = reduce(vcat, itemmap)
	starts = vcat(0, accumulate(+,map(m -> length(m.first.mask), masks))[1:end-1])
	FlatView(tuple(masks...), itemmap,starts)
end

function Base.setindex!(m::FlatView, v, i) 
	j = m.itemmap[i]
	m.masks[j.maskid].first.mask[j.innerid] = v
end

function Base.getindex(m::FlatView, i) 
	j = m.itemmap[i]
	only(unique(m.masks[j.maskid].first.mask[j.innerid]))
end

Base.length(m::FlatView) = length(m.itemmap)
Base.fill!(m::FlatView, v) = map(x -> fill!(x, v), m)

function parent(m::FlatView, i) 
	j = m.itemmap[i]
	parent_id = m.masks[j.maskid].second
	parent_id == 0 && return(0)
	m.starts[parent_id] + index_in_parent(m.masks[j.maskid].first, j.innerid)
end

function Base.map(f, m::FlatView)
	vcat(map(x -> f(x.first), m.masks)...)
end

useditems(m::FlatView) = findall(map(i -> m[i], 1:length(m)))