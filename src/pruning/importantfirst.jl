using Random

function importantfirst!(f, fv::FlatView, significance::Vector{T}; rev::Bool = true, participateonly::Bool = false, oscilate::Bool = false) where {T<:Number}
	ii = participateonly ? sortindices(findall(participate(fv)), significance, rev = rev) : sortperm(significance, rev = rev)
	fill!(fv, false)
	addminimum!(f, fv, significance, ii, strict_improvement = false)
	used = useditems(fv)
	@info "importantfirst: output = $(f()) added $(length(used)) features"
	@assert all(fv[used])
	used_old = used
	while true
		removeexcess!(f, fv, shuffle(used))
		used = useditems(fv)
		length(used) == length(used_old) && break
		used_old = used
	end
	@info "importantfirst: output = $(f()) keeping $(length(used)) features"
	oscilate && oscilate!(f, fv)
end

function importantfirst!(f, ms::AbstractExplainMask, scorefun; rev::Bool = true, oscilate::Bool = false)
	fv = FlatView(ms)
	significance = map(scorefun, fv)
	importantfirst!(f, fv, significance, rev = rev, oscilate = oscilate)
end

function sortindices(ii::Vector{Int}, significance::Vector{T}; rev = true) where {T<:Number}
	is = sort(ii, rev = rev, lt = (i,j) -> significance[i] < significance[j]);
end