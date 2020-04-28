using Random
# function importantfirst!(f, flatmask::FlatView, significance::Vector{T}) where {T<:Number}
# 	fill!(flatmask, false)
# 	fval = f()
# 	@info "output on empty sample = $(fval)"
# 	fval == 0 && return()
# 	ii = sortperm(significance, rev = true);
# 	@show length(ii)
# 	@show length(participate(flatmask))
# 	ii = ii[participate(flatmask)]
# 	addminimum!(f, flatmask, significance, ii, strict_improvement = false)
# 	used = useditems(flatmask)
# 	@info "output = $(f()) added $(length(used)) features"
# 	removeexcess!(f, flatmask, Random.shuffle(ii[used]))
# 	used = useditems(flatmask)
# 	@info "keeping $(length(used)) features"
# 	removeexcess!(f, flatmask, Random.shuffle(ii[used]))
# 	used = useditems(flatmask)
# 	@info "keeping $(length(used)) features"
# end

function importantfirst!(f, fv::FlatView, significance::Vector{T}; participateonly = false) where {T<:Number}
	ii = participateonly ? sortindices(findall(participate(fv)), significance, rev = true) : sortperm(significance, rev = true)
	fill!(fv, false)
	addminimum!(f, fv, significance, ii, strict_improvement = false)
	used = sortindices(useditems(fv), significance, rev = false)
	@info "output = $(f()) added $(length(used)) features"
	@assert all(fv[used])
	removeexcess!(f, fv, used)
	used = useditems(fv)
	@info "keeping $(length(used)) features"
end

function importantfirst!(f, ms::AbstractExplainMask, scorefun)
	fv = FlatView(ms)
	significance = map(scorefun, fv)
	importantfirst!(f, fv, significance)
end

function sortindices(ii::Vector{Int}, significance::Vector{T}; rev = false) where {T<:Number}
	is = sort(ii, rev = true, lt = (i,j) -> significance[i] < significance[j]);
end