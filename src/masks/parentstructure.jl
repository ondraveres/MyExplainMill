"""
	parent_structure(ms::AbstractExplainMask, level = 1)

	recursively traverse the hierarchical mask structure and 
	collect mask that have explainable content and identifies their level.
"""
function parent_structure(ms::AbstractExplainMask, level = 1)
	res = ((mask(ms) === nothing) || isempty(mask(ms))) ? nothing : ms => level
	level = res == nothing ? level : level + 1
	cs = collect(children(ms))
	isempty(cs) && return(res)
	c = map(cs) do n 
		parent_structure(n, level)
	end
	c = vcat(res, c...)
	c = filter(!isnothing, c)
	isempty(c) ? nothing : c
end
