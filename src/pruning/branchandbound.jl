# EMULATE INPUT SET OF SHAPLEY VALUES AND PARENT-CHILD DEPENDENCY OF FEATURES
# constraints of feature tree structure represented by lists of ascendants
# 1
# 3 - 4
#   L 5
# 2 - 6
#   L 7 - 9
#       L 10
#   L 8
dep = Dict{Int,Pair{Float64,Array{Int}}}()
dep[1] = 1.05 => []
dep[2] = 2.95 => []
dep[4] = 4.78 => [3]
dep[3] = 3.15 => []
dep[5] = 5.05 =>  [3]
dep[6] = 8.15 => [2]
dep[7] = 7.05 => [2]
dep[8] = 10.05 => [2]
dep[9] = 6.45 =>  [7, 2]
dep[10] = 9.05 => [7, 2]
dep[11] = 17.9 => [9, 3, 1]
dep[12] = 5.56 => [1]
dep[13] = 4.12 => []
dep[14] = 13.93 => [6, 9, 1]
dep[15] = 19.09 => [10]
dep[16] = 7.65 => [1, 3, 5, 7]
for i in [17:1:1000;]
    dep[i] = i => []
end
# EMULATE CRITERION FUNCTION CALL FOR SUBSET
function JHdemo(subc)
    sleep(0.001)  #0.001=1ms Julia minimum. In real GVMA TP has 0,5ms
    result = 0.0
    ds = 0.0
    for i in 1:length(subc)
        result += subc[i] == 0 ? 0 : dep[i].first
        ds += dep[i].first
    end
    return result / ds
end
@time res = exhaust_explain(JHdemo, 0.5, 7, dep, 4, 60000, true)


# HELPER CODE NEEDED in exhaust_explain
@enum Flow forward evaluate backward stop
function update_subset(subc, constraints, feature, coeff)
    subc[feature] += coeff
    for j in constraints[feature].second
        subc[j] += coeff
    end
end
function next_branch(buf, D, i, sub, subc, pivot, pivotc, constraints)
    nextpivot = pivot[i]+1
    while (i>1) && (pivotc[i,buf[nextpivot]] == 0) && (nextpivot <= pivot[i-1]-1)
        nextpivot += 1
    end
    if i>1 ? nextpivot <= pivot[i-1]-1 : nextpivot <= D
        update_subset(subc,constraints,buf[pivot[i]],-1)
        pivot[i] = nextpivot
        if i>1
            for j in constraints[buf[pivot[i]]].second
                pivotc[i,j] = 0
            end
        end
        sub[i] = buf[pivot[i]]
        update_subset(subc,constraints,buf[pivot[i]],1)
        flow = evaluate
    else
        flow = backward
    end
    return flow
end
# MAIN SEARCH PROCEDURE
# JH - criterion function to evaluate each generated subset
# JHthreshold - keep explanations for which JH(subset)>JHthreshold. Default: 0.5
# constraints - feature shapley values + lists of ascendants
# D - dimensionality
# max_depth - ignore search branches deeper than this. can be used to limit computational time
# max_evals - hard limit on number of evaluations. can be used to limit computational time. Very roughly 60000 ~ 5min
# verbose - println each found explanation during search and println summary at end
function exhaust_explain(JH::Function, JHthreshold, D, constraints::Dict{Int,Pair{Float64,Array{Int}}}, max_depth, max_evals, verbose::Bool = false)
    buf = first.(sort!(collect(filter(p->p.first <= D, constraints)), by = x -> x[2], rev = true))
    pivot = zeros(Int, D)
    pivotc = ones(Int, D, D)
    sub = zeros(Int, D)
    subc = zeros(Int, D)
    explanations = Dict{Array, Float64}()
    counter_found = 0
    counter_all = 0
    i = 1
    pivot[i] = 1
    sub[i] = buf[pivot[i]]
    update_subset(subc,constraints,buf[pivot[i]],1)
    flow::Flow = evaluate
    while flow != stop
        if flow == evaluate
            JHval = JH(subc)
            if JHval > JHthreshold
                if verbose println(sub, "  ", subc, "  ", round(JHval,digits=5)) end
                push!(explanations,[subc[i] == 0 ? 0 : 1 for i in 1:D] => JHval)
                counter_found += 1
                flow = next_branch(buf, D, i, sub, subc, pivot, pivotc, constraints)
            elseif pivot[i] == 1
                flow = next_branch(buf, D, i, sub, subc, pivot, pivotc, constraints)
            elseif i<D && i<max_depth
                flow = forward
            else
                flow = backward
            end
            counter_all += 1
            if(counter_all > max_evals)
                if verbose println("Terminated at ", counter_all, " evals limit.") end
                flow = stop
            end
        elseif flow == forward
            i += 1
            pivot[i] = 1
            for j in 1:D pivotc[i,j] = 1 end
            for j in constraints[buf[pivot[i]]].second
                pivotc[i,j] = 0
            end
            sub[i] = buf[pivot[i]]
            update_subset(subc,constraints,buf[pivot[i]],1)
            flow = evaluate
        elseif flow == backward
            sub[i] = 0  #keep to ease debugging
            update_subset(subc,constraints,buf[pivot[i]],-1)
            i -= 1
            if i > 0
                flow = next_branch(buf, D, i, sub, subc, pivot, pivotc, constraints)
            else
                flow = stop
            end
        end
    end
    if verbose println("Explanations found: ", counter_found, ", reduced: ", length(explanations)) end
    return explanations
end
