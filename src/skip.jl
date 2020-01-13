# The SkipDaf is effectively a noop for explanatio

struct SkipDaf end
struct EmptyMask end

# Duff.Daf(::PathNode) = SkipDaf()

StatsBase.sample(daf::SkipDaf, ds) = (ds, EmptyMask())

Duff.update!(daf::SkipDaf, mask::EmptyMask, v::Number, valid_columns = nothing) = nothing

prune(ds, mask::EmptyMask) = ds

masks_and_stats(daf::SkipDaf, depth = 0) = (EmptyMask(), [])

Mill.dsprint(io::IO, n::SkipDaf; pad=[]) = paddedprint(io, "SkipDaf")