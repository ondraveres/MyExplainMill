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
