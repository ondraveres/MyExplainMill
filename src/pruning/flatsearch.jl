using Random

function flatsearch!(f, fv::FlatView; rev::Bool = true, participateonly::Bool = false, fine_tuning::Bool = false, max_n = 5, random_removal::Bool = true) where {T<:Number}
	significance = heuristic(fv)
	ii = participateonly ? sortindices(findall(participate(fv)), significance, rev = rev) : sortperm(significance, rev = rev)
	fill!(fv, false)
	addminimumbi!(f, fv, significance, ii)
	used = useditems(fv)
	@debug "flatsearch: output = $(f()) added $(length(used)) features"
	random_removal && randomremoval!(f, fv)
	used = useditems(fv)
	@debug "flatsearch: output = $(f()) keeping $(length(used)) features"
	fine_tuning && finetune!(f, fv, max_n)
end
flatsearch!(f, mk::AbstractStructureMask; kwargs...) = flatsearch!(f, FlatView(mk); participateonly = false, kwargs...)

function flatsfs!(f, fv::FlatView; fine_tuning::Bool = false, participateonly::Bool = false, max_n = 5, random_removal::Bool = true)
	@debug "sfs: length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
	sfs!(f, fv)
	used = useditems(fv)
	@debug "flatsearch: output = $(f()) added $(length(used)) features"
	random_removal && randomremoval!(f, fv)
	used = useditems(fv)
	@debug "flatsearch: output = $(f()) keeping $(length(used)) features"
	fine_tuning && finetune!(f, fv, max_n)
end


function flatsfs!(f, mk::AbstractStructureMask; kwargs...) 
	flatsfs!(f, FlatView(add_participation(mk));kwargs...)
end

function sortindices(ii::Vector{Int}, significance::Vector{T}; rev = true) where {T<:Number}
	is = sort(ii, rev = rev, lt = (i,j) -> significance[i] < significance[j]);
end
