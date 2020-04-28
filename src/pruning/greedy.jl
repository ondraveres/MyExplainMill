function greedy!(f, fv::FlatView, significance::Vector{T}) where {T<:Number}
	fill!(fv, false)
	fval = f()
	@info "output on empty sample = $(fval)"
	fval == 0 && return()
	ii = sortperm(significance, rev = true);
	addminimum!(f, fv, significance, ii, strict_improvement = false)
	used = useditems(fv)
	@info "Explanation uses $(length(used)) features out of $(length(fv))"
	@info "output on explaination should be zero = $(f())"
end

function greedy!(f, ms::AbstractExplainMask, scorefun)
	fv = FlatView(ms)
	significance = map(scorefun, fv)
	greedy!(f, fv, significance)
end