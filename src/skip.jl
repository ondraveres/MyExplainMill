# The SkipDaf is effectively a noop for explanatio

struct EmptyMask <:AbstractExplainMask end

function StatsBase.sample!(pruning_mask::EmptyMask)

end

mask(::EmptyMask) = Vector{Bool}()

participate(::EmptyMask) = Vector{Bool}()

prune(ds, mask::EmptyMask) = ds

mapmask(f, mask::EmptyMask) = nothing

invalidate!(mask::EmptyMask, observations::Vector{Int}) = nothing
