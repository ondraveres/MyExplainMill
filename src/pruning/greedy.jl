function greedy!(f, fv::FlatView, significance::Vector{T}, k::Int) where {T<:Number}
	fill!(fv, false)
	@info "output on empty sample = $(fval)"
	ii = sortperm(significance, rev = true);
	for i in ii[1:k]
		fv[i] = true
	end
	used = useditems(fv)
	@info "Explanation uses $(length(used)) features out of $(length(fv))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end

"""
	greedy!(f, ms::AbstractExplainMask, scorefun, k...)
	greedy!(f, fv::FlatView, significance::Vector{T}, k...)

	selects first `k` items. If `k` is not set, it is determined such that the output is positive
"""	
function greedy!(f, fv::FlatView, significance::Vector{T}) where {T<:Number}
	fill!(fv, false)
	fval = f()
	@info "output on empty sample = $(fval)"
	fval == 0 && return()
	ii = sortperm(significance, rev = true);
	addminimum!(f, fv, significance, ii, strict_improvement = false)
	used = useditems(fv)
	@info "Explanation uses $(length(used)) features out of $(length(fv))"
	f() < 0 && @error "output of explaination is $(f()) and should be zero"
end

function greedy!(f, ms::AbstractExplainMask, scorefun, k...)
	fv = FlatView(ms)
	significance = map(scorefun, fv)
	greedy!(f, fv, significance, k...)
end