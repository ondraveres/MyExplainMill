"""
	A structure presenting masks in the PruiningMask structure
	as a flat array of coefficients, ideally sorted according to 
	chosen value. Hence, the user does not have to be aware of the structure
	tree structure, which is abstracted as flat array. The structure supports
	indexing through `setindex` and `getindex` and it also returns the parent 
	of an item in `parent`, which points to an item which can influence the given 
	point.
"""
struct FlatView{V, I}
	masks::V
	itemmap::Vector{I}
	starts::Vector{Int}
end

"""
	FlatView(mk::AbstractStructureMask)
	FlatView(masks::Vector)

	Create a FlatView from a vector of masks or from an `mk::AbstractStructureMask`.
	If `FlatView` is constructed `AbstractStructureMask`, the itemmap contains 
	indication of a level (distance from the root) of the mask. If constructed from 
	a vector of masks, the level is all zeros. 
"""
function FlatView(mk::AbstractStructureMask)
	flatview(collect_masks_with_levels(mk))
end

function FlatView(masks::Vector)
	flatview(map(m -> m => 0, masks))
end

function FlatView(masks::Vector{<:Pair})
	flatview(masks)
end

function flatview(mask_pairs::Vector)
	itemno = 0
	itemmap = map(enumerate(mask_pairs)) do (i, smk)
		m, l = smk
		map(1:length(m)) do j
			itemno += 1
			(itemid = itemno, maskid = i, innerid = j, level = l)
		end
	end
	masks = map(first, mask_pairs)
	itemmap = reduce(vcat, itemmap)
	starts = vcat(0, accumulate(+,map(m -> length(m), masks))[1:end-1])
	FlatView(tuple(masks...), itemmap, starts)
end

function Base.setindex!(m::FlatView, v, i::Int) 
	mi = m.itemmap[i] 
	m.masks[mi.maskid][mi.innerid] = v
end

function Base.setindex!(m::FlatView, v, ii::Vector{Int}) 
	foreach(i -> m[i] = v, ii)
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
	mi = m.itemmap[i] 
	m.masks[mi.maskid][mi.innerid]
end

function Base.getindex(m::FlatView, ii::Vector{Int}) 
	map(i -> m[i], ii)
end


Base.fill!(fv::FlatView, v) = foreach(i -> fv[i] = v, 1:length(fv))
Base.ndims(::Type{<:FlatView}) = 1
Base.length(m::FlatView) = length(m.itemmap)
Base.size(fv::FlatView) = (length(fv),)
Base.dotview(fv::FlatView, ii) = error("Dotview not overloaded for FlatView")

useditems(m::FlatView) = findall(usedmask(m))
usedmask(m::FlatView) = map(i -> m[i], 1:length(m)) #this is really ineffective but that is life
participate(m::FlatView) = reduce(vcat, map(participate, m.masks))
support_participation(m::FlatView) = all(support_participation(i) for i in m.masks)
heuristic(m::FlatView) = reduce(vcat, map(heuristic, m.masks))
copy2vec(m::FlatView) = reduce(vcat, map(prunemask, m.masks))
