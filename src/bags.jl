struct BagDaf{C,B,M} <: AbstractDaf
	child::C
	bags::B
	daf::M
end

struct BagMask{C,M}
	child_masks::C
	mask::M
end

mask_length(ds::BagNode) = nobs(ds.data)

function StatsBase.sample(daf::BagDaf, ds::BagNode{Missing,AlignedBags,Nothing})
	return(ds, nothing)
end

function StatsBase.sample(daf::BagDaf, ds::BagNode)
	mask = rand(nobs(ds.data)) .>= 0.5
	dss = removeinstances(ds, mask)
	x, child_masks = sample(daf.child, dss.data)
	return(BagNode(x, dss.bags), BagMask(child_masks, mask))
end

function Duff.Daf(ds::BagNode)
	BagDaf(Duff.Daf(ds.data), ds.bags, Duff.Daf(mask_length(ds)))
end

function prune(ds::BagNode, mask::BagMask)
	x = prune(ds.data, mask.child_masks)
	x = Mill.subset(x, findall(mask.mask))
	bags = Mill.adjustbags(ds.bags, mask.mask)
	BagNode(x, bags)
end

Duff.update!(daf::BagDaf, mask::BagMask, v::Number, valid_columns::Nothing) = Duff.update!(daf::BagDaf, mask::BagMask, v::Number)

function Duff.update!(daf::BagDaf, mask::BagMask, v::Number, valid_columns = collect(1:length(daf.bags)))
	valid_bags = daf.bags[valid_columns]
	valid_columns = reduce(vcat, collect.(valid_bags))
	valid_sub_indexes = valid_columns[mask.mask]
	Duff.update!(daf.daf, mask.mask, v, valid_columns)
	Duff.update!(daf.child, mask.child_masks, v, valid_sub_indexes)
end

function masks_and_stats(daf::BagDaf, depth = 0)
	(child_masks, child_dafs) = masks_and_stats(daf.child, depth + 1)
	mask = fill(true, length(daf.daf))
	return(BagMask(child_masks,mask), vcat(child_dafs, (m = mask, d = daf.daf, depth = depth)))
end

function dsprint(io::IO, n::BagDaf; pad=[])
    c = COLORS[(length(pad)%length(COLORS))+1]
    paddedprint(io,"Bag\n", color=c)
    paddedprint(io, "  └── ", color=c, pad=pad)
    dsprint(io, n.child, pad = [pad; (c, "      ")])
end
