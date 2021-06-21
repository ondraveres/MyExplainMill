struct SimpleMask{T}
	x::Vector{T}
end

prunemask(m::SimpleMask{Bool}) = m.x
prunemask(m::SimpleMask{<:Number}) = m.x .> 0
diffmask(m::SimpleMask) = m.x
Base.length(m::SimpleMask) = length(m.x)
Base.getindex(m::SimpleMask, i) = m.x[i]
Base.setindex!(m::SimpleMask, v, i) = m.x[i] = v
Base.materialize!(m::SimpleMask, v) = m.x .= v