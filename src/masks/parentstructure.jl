const ParentStructure = Vector{Pair{AbstractStructureMask, Int}}
"""
	parent_structure(mk::AbstractStructureMask, level = 1)

	recursively traverse the hierarchical mask structure and 
	collect mask that have explainable content and identifies their level.
"""
function parent_structure(mk::AbstractNoMask, level = 1)
	cs = collect(children(mk))
	isempty(cs) && return(ParentStructure())
	cs = map(cs) do n 
		parent_structure(n, level)
	end
	reduce(vcat, cs)
end

function parent_structure(mk::AbstractStructureMask, level = 1)
	cs = collect(children(mk))
	isempty(cs) && return([mk => level])
	cs = map(cs) do n 
		parent_structure(n, level + 1)
	end
	reduce(vcat, [mk => level, cs...])
end
