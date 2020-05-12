struct Terms 
	terms::Dict{Any,Nothing}
end

Terms() = Terms(Dict{Any,Nothing}())
Base.haskey(t::Terms, k) = haskey(t.terms, k)
Base.push!(t::Terms, k) = t[k] = nothing

function complexity(d::Dict)
	ks = collect(keys(d))
	isempty(ks) && return(0)
	if length(ks) == 1 
		k = only(ks)
		k == :or && return(complexity_or(d[:or]))
		k == :and && return(complexity_and(d[:and]))
	end
	1 + mapreduce(k -> complexity(d[k]), +, ks)
end

complexity(vs::Vector) = isempty(vs) ? 0 : mapreduce(complexity, +, vs)
complexity(s::AbstractString) = 1 

complexity_and(vs::Vector{T}) where {T<:AbstractString} = length(vs)
complexity_and(vs::Vector) = mapreduce(complexity, +, vs)
complexity_and(d::Dict{Symbol,String}) = complexity(d)

complexity_or(vs::Vector{T}) where {T<:AbstractString} = 1
complexity_or(vs::Vector) = mapreduce(complexity, +, vs)
# complexity_or(vs::Vector) = maximum(map(complexity, vs))
