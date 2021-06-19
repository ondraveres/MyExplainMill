struct SimpleMask
	x::Vector{Bool}
end

prunemask(m::SimpleMask) = m.x
diffmask(m::SimpleMask) = m.x
Base.length(m::SimpleMask) = length(m.x)
Base.getindex(m::SimpleMask, i) = m.x[i]
Base.setindex!(m::SimpleMask, v, i) = m.x[i] = v
Base.materialize!(m::SimpleMask, v) = m.x .= v