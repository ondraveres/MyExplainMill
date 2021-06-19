using Random

function flatsearch!(f, fv::FlatView, significance::Vector{T}; rev::Bool = true, participateonly::Bool = false, fine_tuning::Bool = false, max_n = 5, random_removal::Bool = true) where {T<:Number}
	ii = participateonly ? sortindices(findall(participate(fv)), significance, rev = rev) : sortperm(significance, rev = rev)
	fill!(fv, false)
	addminimumbi!(f, fv, significance, ii)
	used = useditems(fv)
	@debug "flatsearch: output = $(f()) added $(length(used)) features"
	@assert all(fv[used])
	random_removal && randomremoval!(f, fv)
	used = useditems(fv)
	@debug "flatsearch: output = $(f()) keeping $(length(used)) features"
	fine_tuning && finetune!(f, fv, max_n)
end

function flatsearch!(f, ms::AbstractStructureMask, scorefun; rev::Bool = true, fine_tuning::Bool = false, max_n = 5, random_removal::Bool = true)
	fv = FlatView(ms)
	significance = map(scorefun, fv)
	flatsearch!(f, fv, significance, rev = rev, max_n = max_n, random_removal = random_removal, fine_tuning = fine_tuning)
end


function flatsfs!(f, ms::AbstractStructureMask, scorefun; participateonly::Bool = false, fine_tuning::Bool = false, max_n = 5, random_removal::Bool = true)
	fv = FlatView(ms)
	flatsfs!(f, fv, max_n = max_n, random_removal = random_removal, fine_tuning = fine_tuning)
end

function flatsfs!(f, ms::AbstractStructureMask; participateonly::Bool = false, fine_tuning::Bool = false, max_n = 5, random_removal::Bool = true)
	fv = FlatView(ms)
	flatsfs!(f, fv, max_n = max_n, random_removal = random_removal, fine_tuning = fine_tuning)
end

function flatsfs!(f, fv::FlatView; fine_tuning::Bool = false, max_n = 5, random_removal::Bool = true)
	@debug "sfs: length of mask: $(length(fv)) participating: $(sum(participate(fv)))"
	sfs!(f, fv)
	used = useditems(fv)
	@debug "flatsearch: output = $(f()) added $(length(used)) features"
	@assert all(fv[used])
	random_removal && randomremoval!(f, fv)
	used = useditems(fv)
	@debug "flatsearch: output = $(f()) keeping $(length(used)) features"
	fine_tuning && finetune!(f, fv, max_n)
end

function sortindices(ii::Vector{Int}, significance::Vector{T}; rev = true) where {T<:Number}
	is = sort(ii, rev = rev, lt = (i,j) -> significance[i] < significance[j]);
end
