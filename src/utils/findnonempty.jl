
#find non-empty lenses
""" findnonempty(ds::Mill.AbstractNode)

    array of lenses corresponding to non-empty leafs in the data node
"""
function findnonempty(ds::ArrayNode)
  nobs(ds) == 0  ? nothing : [@lens _.data]
end

function findnonempty(ds::LazyNode)
  nobs(ds) == 0  ? nothing : [@lens _.data]
end

function findnonempty(ds::BagNode)
    childlenses = findnonempty(ds.data)
    isnothing(childlenses) && return(childlenses)
    map(l -> Setfield.PropertyLens{:data}() ∘ l, childlenses)
end

function findnonempty(ds::ProductNode)
  lenses = mapreduce(vcat, keys(ds)) do k 
    childlenses = findnonempty(ds[k])
    isnothing(childlenses) && return(childlenses)
    map(l -> Setfield.PropertyLens{:data}() ∘ Setfield.PropertyLens{k}() ∘ l, childlenses)
  end
  isnothing(lenses) && return(nothing)
  lenses = filter(!isnothing, lenses)
  isempty(lenses) ? nothing : lenses
end