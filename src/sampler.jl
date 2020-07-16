function json_sample(d::Dict)
	ks = collect(keys(d))
	if length(ks) == 1 
		k = only(ks)
		k == :or && return(sample_or(d[:or]))
		k == :and && return(sample_and(d[:and]))
	end
	Dict(map(k -> k => json_sample(d[k]), ks))
end

json_sample(s::String) = s 
json_sample(s::Number) = s 
sample_or(vs::Vector) = json_sample(sample(vs))
sample_and(vs::Vector) = map(json_sample, vs)
json_sample(vs::Vector) = map(json_sample, vs)