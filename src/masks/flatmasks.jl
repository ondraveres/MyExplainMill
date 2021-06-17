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

function FlatView(mk::AbstractExplainMask)
	FlatView(map(first, parent_structure(mk)))
end

function FlatView(masks::Vector{<:Pair})
	FlatView(map(first, masks))
end

function FlatView(masks::Vector)
	itemno = 0
	itemmap = map(enumerate(masks)) do (i,m)
		map(1:length(m.mask)) do j
			itemno += 1
			(itemid = itemno, maskid = i, innerid = j)
		end
	end
	itemmap = reduce(vcat, itemmap)
	starts = vcat(0, accumulate(+,map(m -> length(m.mask), masks))[1:end-1])
	FlatView(tuple(masks...), itemmap,starts)
end

function Base.setindex!(m::FlatView, v, i::Int) 
	j = m.itemmap[i]
	m.masks[j.maskid].mask[j.innerid] = v
end


"""
	settrue!(fv::FlatView, bitmask::Vector{Bool})

	sets masks in `fv` to bitmask ignoring all dependencies
"""
function settrue!(fv::FlatView, bitmask::Vector{Bool})
	for i in 1:length(fv.masks)
		m = ExplainMill.prunemask(fv.masks[i])
		rg = fv.starts[i]+1:fv.starts[i]+length(m)
		m .= bitmask[rg]
	end
end

# this is a syntactic sugar such that I can write fv .= x
function Base.materialize!(fv::FlatView, a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1},Nothing,typeof(identity),Tuple{Array{Bool,1}}} )
	settrue!(fv, a.args[1])
end

function Base.getindex(m::FlatView, i::Int) 
	j = m.itemmap[i]
	only(unique(m.masks[j.maskid].mask[j.innerid]))
end

function Base.getindex(m::FlatView, ii::Vector{Int}) 
	map(i -> m[i], ii)
end

Base.length(m::FlatView) = length(m.itemmap)
Base.fill!(m::FlatView, v) = map(x -> fill!(x.mask.mask, v), m.masks)

function parent(m::FlatView, i) 
	j = m.itemmap[i]
	parent_id = m.masks[j.maskid].second
	parent_id == 0 && return(0)
	m.starts[parent_id] + index_in_parent(m.masks[j.maskid], j.innerid)
end

function Base.map(f, m::FlatView)
	vcat(map(x -> f(x), m.masks)...)
end

useditems(m::FlatView) = findall(usedmask(m))
usedmask(m::FlatView) = map(i -> m[i], 1:length(m))
participate(m::FlatView) = reduce(vcat, map(i -> participate_item(i.mask), m.masks))
