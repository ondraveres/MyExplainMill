function removeexcess!(f, flatmask::FlatView, x::Vector{Int})
	previous =  f()
	previous < 0 && return(false)
	changed = false
	for i in useditems(flatmask)
		flatmask[i] == false && continue
		flatmask[i] = false
		o = f()
		if o < 0
			flatmask[i] = true
		end
		changed = true
	end
	return(changed)
end

"""
	addminimum!(f, flatmask, significance, ii; strict_improvement::Bool = true)

	turns on first `k` subtrees (also called features or how you like it)
	such that the output of `f` is greater or equal than zero. The search 
	is performed linearly from start to `k`, which means that complexity
	is `O(n)`, where `n` is length of `ii`.
	ii --- are indexes that are flipped in `flatmask` during the search
"""
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

function setflatmask!(flatmask, ii, mid)
	foreach(i -> flatmask[ii[i]] = true, 1:mid)
	foreach(i -> flatmask[ii[i]] = false, mid + 1:length(ii))
end

"""
	addminimumbi!(f, flatmask, significance, ii; strict_improvement::Bool = true)

	turns on first `k` subtrees (also called features or how you like it)
	such that the output of `f` is greater or equal than zero. The search 
	is performed by bisection, which means that the complexity
	is `O(log(n))`, where `n` is length of `ii`.
	ii --- are indexes that are flipped in `flatmask` during the search
"""
function addminimumbi!(f, flatmask, significance, ii)
	previous =  f()
	previous > 0 && return(false)
	right = length(ii)
	left = 1
	while true
		# @show (left, right, right - left)
		right - left <= 1 && break
		mid = left + div(right - left, 2)
		setflatmask!(flatmask, ii, mid)
		o = f()
		if o >= 0
			right = mid 
		else
			left = mid
		end
	end
	setflatmask!(flatmask, ii, right)
	return(true)
end

"""
	addone!(f, flatmask)

	adds one feature which maximally improves the value without taking the importance in the account
"""
function addone!(f, flatmask)
	o = f()
	best, j =  typemin(o), -1
	ii = setdiff(findall(participate(flatmask)), useditems(flatmask))
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
	removeone!(f, flatmask, ii = ExplainMill.useditems(flatmask))

	removes one feature which least decreased the value without taking the importance in the account
"""
function removeone!(f, flatmask, ii = ExplainMill.useditems(flatmask))
	o = f()
	best, j =  typemin(o), -1
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
	@debug "finetune: output = $(f()) keeping $(length(useditems(flatmask))) features"
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
		if f() >= 0
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


"""
	greedyremoval!(f, flatmask)

	greedily remove items such that `f` uses minimal number 
	of items and it is above zero. In each iteration, it removes
	the item causing least decrease of `f`
"""
function greedyremoval!(f, flatmask)
	while true
		ii = useditems(flatmask)
		changed = removeone!(f, flatmask, ii)
		!changed && break
		if f() < 0 
			foreach(i -> flatmask[i] = true, ii)
			break
		end
	end
end
