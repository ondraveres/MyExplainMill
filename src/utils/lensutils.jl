
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
    map(l -> Setfield.PropertyLens{:data}() ∘ (Setfield.PropertyLens{k}() ∘ l), childlenses)
  end
  isnothing(lenses) && return(nothing)
  lenses = filter(!isnothing, lenses)
  isempty(lenses) ? nothing : lenses
end

function ModelLens(model::ProductModel, lens::Setfield.ComposedLens)
  if lens.outer == @lens _.data
    return(Setfield.PropertyLens{:ms}() ∘ ModelLens(model.ms, lens.inner))
  end
  return(lens)
end

function ModelLens(model, lens::Setfield.ComposedLens)
  outerlens = ModelLens(model, lens.outer)
  outerlens ∘ ModelLens(get(model, outerlens), lens.inner)
end

function ModelLens(::BagModel, lens::Setfield.PropertyLens{:data})
  @lens _.im
end

function ModelLens(::NamedTuple, lens::Setfield.PropertyLens)
  lens
end

function ModelLens(::ProductModel, lens::Setfield.PropertyLens{:data})
  @lens _.ms
end

function ModelLens(::ArrayModel, ::Setfield.PropertyLens{:data})
 @lens _.m
end


