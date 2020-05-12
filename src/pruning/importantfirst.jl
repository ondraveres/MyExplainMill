using Random

function importantfirst!(f, fv::FlatView, significance::Vector{T}; participateonly = false) where {T<:Number}
	ii = participateonly ? sortindices(findall(participate(fv)), significance, rev = true) : sortperm(significance, rev = true)
	fill!(fv, false)
	addminimum!(f, fv, significance, ii, strict_improvement = false)
	# used = sortindices(useditems(fv), significance, rev = false)
	used = useditems(fv)
	@info "output = $(f()) added $(length(used)) features"
	@assert all(fv[used])
	used_old = used
	while true
		removeexcess!(f, fv, shuffle(used))
		used = useditems(fv)
		length(used) == length(used_old) && break
		@info "output = $(f()) keeping $(length(used)) features"
		used_old = used
	end
end

function importantfirst!(f, ms::AbstractExplainMask, scorefun)
	fv = FlatView(ms)
	significance = map(scorefun, fv)
	importantfirst!(f, fv, significance)
end

function sortindices(ii::Vector{Int}, significance::Vector{T}; rev = false) where {T<:Number}
	is = sort(ii, rev = true, lt = (i,j) -> significance[i] < significance[j]);
end