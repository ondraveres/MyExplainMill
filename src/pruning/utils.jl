function removeexcess!(f, flatmask, ii)
	@debug "enterring removeexcess"
	previous =  f()
	previous < 0 && return(false)
	changed = false
	for i in ii
		flatmask[i] == false && continue
		flatmask[i] = false
		o = f()
		# @show (i,o)
		if o < 0
			flatmask[i] = true
		end
		changed = true
	end
	return(changed)
end

function addminimum!(f, flatmask, significance, ii; strict_improvement::Bool = true)
	changed = false
	previous =  f()
	previous > 0 && return(changed)
	for i in ii
		all(flatmask[i]) && continue
		flatmask[i] = true
		o = f()
		if strict_improvement && o <= previous
			flatmask[i] = false
		else 
			previous = o
			changed = true
		end
		if o >= 0
			break
		end
	end
	return(changed)
end

function addminimumbi!(f, flatmask, significance, ii; strict_improvement::Bool = true)
	changed = false
	previous =  f()
	previous > 0 && return(changed)
	for i in ii
		all(flatmask[i]) && continue
		flatmask[i] = true
		o = f()
		if strict_improvement && o <= previous
			flatmask[i] = false
		else 
			previous = o
			changed = true
		end
		if o >= 0
			break
		end
	end
	return(changed)
end

"""
	addone!(f, flatmask)

	adds one feature which maximally improves the value without taking the importance in the account
"""
function addone!(f, flatmask)
	o = f()
	best, j =  typemin(o), -1
	ii = setdiff(1:length(flatmask), ExplainMill.useditems(flatmask))
	for i in ii
		all(flatmask[i]) && continue
		flatmask[i] = true
		o = f()
		flatmask[i] = false
		if o > best 
			best, j = o, i
		end
	end

	if j != -1
		flatmask[j] = true
		return(true)
	end
	return(false)
end

"""
	removeone!(f, flatmask)

	removes one feature which least decreased the value without taking the importance in the account
"""
function removeone!(f, flatmask)
	o = f()
	best, j =  typemin(o), -1
	ii = ExplainMill.useditems(flatmask)
	for i in ii
		!all(flatmask[i]) && continue
		flatmask[i] = false
		o = f()
		flatmask[i] = true
		if o > best 
			best, j = o, i
		end
	end

	if j != -1
		flatmask[j] = false
		return(true)
	end

	return(false)
end


function settobest!(fv, visited_states::Dict{K,V}) where {K,V}
	isempty(visited_states) && return()
	iis = collect(keys(visited_states))
	iis = filter(i -> visited_states[i] >= 0, iis)
	minl = minimum(length.(iis))
	iis = filter(i -> length(i) == minl, iis)
	ii, b = iis[1], visited_states[iis[1]]
	for i in iis
		if b > visited_states[i]
			ii, b = i, visited_states[i]
		end
	end
	# @info "selecting $(length(ii)) with $(visited_states[ii])"

	for i in 1:length(fv)
		fv[i] = i ∈ ii
	end
end


function finetune!(f, flatmask, max_n = typemax(Int))
	visited_states = Dict(sort(useditems(flatmask)) => f())
	n, max_n = 1, min(max_n, length(useditems(flatmask)))
	for i in 1:100
		if mod(i, 2) == 0
			finetuneadd!(f, flatmask, n)
		else
			finetuneremove!(f, flatmask, n)
		end
		ii = sort(useditems(flatmask))

		if haskey(visited_states, ii)
			n += 1 
			n > max_n && break 
		else
			visited_states[ii] = f()
			max_n = min(length(ii), max_n)
			# summarizestats(visited_states)
		end
	end
	settobest!(flatmask, visited_states)
	@info "finetune: output = $(f()) keeping $(length(useditems(flatmask))) features"
end

function finetuneadd!(f, flatmask, n::Int)
	for _ in 1:n 
		addone!(f, flatmask)
	end

	while true
		!removeone!(f, flatmask) && break
		if f() < 0
			addone!(f, flatmask)
			break
		end
	end
end

function finetuneremove!(f, flatmask, n::Int)
	for _ in 1:n 
		removeone!(f, flatmask)
	end

	f() > 0 && return

	while true
		!addone!(f, flatmask)
		if f() > 0
			break
		end
	end
end

function sfs!(f, flatmask; random_removal::Bool = false)
	fill!(flatmask, false)
	while f() < 0 
		!addone!(f, flatmask) && break
	end

	random_removal && randomremoval!(f, flatmask)
end


function randomremoval!(f, flatmask)
	used_old = useditems(flatmask)
	while true
		removeexcess!(f, flatmask, shuffle(used_old))
		used = useditems(flatmask)
		length(used) == length(used_old) && break
		used_old = used
	end
end