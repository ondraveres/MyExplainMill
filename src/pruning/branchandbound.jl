function functionalize(f, fv)
    function ff(x)
        settrue!(fv, x)
        f() 
    end 
    ff, fill(false, length(fv))
end

function branchandbound!(feval, fv::FlatView, significance; max_funevals = 1_000_000, max_time = 600)
    valid_indexes = findall(participate(fv))
    f, x = functionalize(feval, fv)
    best_fval, best_x = f(x), x 
    println("Branch and bound in a solution vector of length ", length(best_x))
    println("initial solution: objective = ", best_fval, " length = ", sum(best_x))
    funevals = 0
    queue = PriorityQueue{typeof(x), typeof(best_fval)}(Base.Order.Reverse)
    enqueue!(queue, deepcopy(x), best_fval)
    evaluated = Dict{typeof(x), typeof(best_fval)}()
    start_time = time()
    while !isempty(queue)
        time() - start_time > max_time && break
        x = dequeue!(queue)

        # If we have a solution and `x` is longer than our current best, 
        # it cannot be optimal
        if best_fval > 0 && sum(x) > sum(best_x) 
            evaluated[x] = typemin(eltype(x))
            continue
        end

        funevals += 1
        funevals > max_funevals && break

        mod(funevals, 10000) == 0 && println("funevals: ",funevals," size of the queue: ", length(queue))

        fval = f(x)
        evaluated[x] = fval

        # update the best solution found so far
        best_fval, best_x = better_solution(best_fval, best_x, fval, x)
        if best_x == x
            println("evaluations = ",funevals, " items = ", sum(best_x), " objective = ",best_fval)
            # If the solution is above threshold, we can try to prune it (which is cheap) to find better
            if best_fval > 0 
                best_fval, best_x = removeexcess!(f, x)
            end
        end

        !promissing(best_fval, best_x, fval, x) && continue
        
        #enque all childrens
        for x in childrens(x, valid_indexes, fval, significance)
            haskey(queue, x[1]) && continue
            haskey(evaluated, x[1]) && continue
            enqueue!(queue, x...)
        end
    end
    println("returned solution: objective = ", f(best_x), " length = ", sum(best_x))
    evaluated
end

"""
    function removeexcess!(f, fv::FlatView, x::Vector{Bool})

    tries to remove superfluous items from `x` but while keeping
    f(x) above zero
"""
function removeexcess!(f, x₀::Vector{Bool})
    f₀ =  f(x₀)
    x = deepcopy(x₀)
    f₀ < 0 && return(false)
    n = sum(x)
    while true 
        for i in findall(x)
            x[i] = false 
            fval = f(x)
            if fval < 0
                x[i] = true
            end
        end
        n == sum(n) && break
        n = sum(n)
    end
    fval = f(x)
    if sum(x) < sum(x₀)
        println("pruned to items = ", sum(x), " objective = ",fval)
    end
    better_solution(fval, x, f₀, x₀)
end

"""
    function promissing(best_fval, best_x, fval, x)

    if best solution `best_fval` is above zero (acceptable), 
    then `x` is promissing only if it is shorter
"""
function promissing(best_fval, best_x, fval, x)
    best_fval > 0 && return(sum(x) < sum(best_x))   
    return(true)
end

"""
    best_fval, best_x = better_solution(best_fval, best_x, fval, x)

    returns better solution, such that the final solution is the shortest 
    above the threshold and among those with the same length,
    we want the one with highest objective value.

    The complicated logic goes as follows.
    If the best solution is not above zero (the threshold), we return 
    the  solution with higher objective value or shorter if equal.
    If the best solution is above zero, then we either take shorter if 
    the proposed is above zero as well, or if they have the same length, 
    we take one with higher objective value. 
"""
function better_solution(best_fval, best_x, fval, x)
    # if the best solution so far is below zero, we take 
    # solution with higher fval or shorter (if equals)
    if best_fval <= 0
        if best_fval < fval
            return(fval, x)
        elseif best_fval == fval && sum(best_x) > sum(x)
            return(fval, x)
        else 
            return(best_fval, best_x)
        end
    # otherwise, we take the shorter solution if fval > 0
    else
        if fval < 0 
            return(best_fval, best_x)
        elseif sum(best_x) > sum(x)     #select the shorter one
            return(fval, x)
        elseif (sum(best_x) == sum(x))  #if equal length, select the better
            if best_fval > fval
                return(best_fval, best_x)
            else 
                return(fval, x)
            end
        else 
           return(best_fval, best_x)
        end
    end
end

function childrens(x::Vector{Bool}, valid_indexes, fval, significance::Nothing)
    map(intersect(findall(.!x),valid_indexes)) do i 
        xx = deepcopy(x)
        xx[i] = true
        (xx, fval)
    end
end

function childrens(x::Vector{Bool}, valid_indexes, fval, significance)
    map(intersect(findall(.!x),valid_indexes)) do i 
        xx = deepcopy(x)
        xx[i] = true
        (xx, fval + significance[i])
    end
end

"""
    function isuseless(useless::Dict{Vector{Int}, <:Number}, x)

    if the function `f` is monotone, which means that adding a term increases 
    and objective value, then we know that if `[a,b,c]` is not a solution 
    (the objective is below zero), then noen of its subsets, 
    i.e. [a], [b], [c], [a,b],[a,c], [b,c]` is not a solution either. 
    This filter implements the check, where dict `useless` keeps all useless
    solutions
"""
function isuseless(useless::Dict{Vector{Int}, <:Number}, x)
    xi = findall(x)
    ks = filter(k -> xi ⊆ k, keys(useless))
    if !isempty(ks)
        # ks = collect(ks)
        # fv .= x 
        # fval = f()
        # if fval > useless[ks[1]]
        #     println(xi," with ",fval," is subset of ",ks[1], " with f = ", useless[ks[1]])
        #     evaluated[x] = typemin(eltype(x))
        # end
        # continue
    end
end
