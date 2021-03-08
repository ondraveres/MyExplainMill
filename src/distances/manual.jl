"""
    evaluatesubstitution(dst_ds, src_ds, model, oids)

    outputs of `model` at outputs `oids` on sample `dst_ds`, where leafs are 
    replaced by corresponding leafs from `src_ds`
"""
function evaluatesubstitution(dst_ds, src_ds, model, oids)
  mapreduce(vcat, findnonempty(dst_ds)) do lens 
    evaluatesubstitution(dst_ds, src_ds, model, oids, lens)
  end
end

function evaluatesubstitution(dst_ds, src_ds, model, oids, lens)
  test_ds = deepcopy(dst_ds)
  src_x = get(src_ds, lens)
  dst_x = get(test_ds, lens)
  o = evaluatesubstitution(test_ds, dst_x, src_x, model, oids)
  o[:lens] = fill(lens, size(o,1))
  o
end


function evaluatesubstitution(test_ds, dst_x::NGramMatrix, src_x::NGramMatrix, model, oids)
  mapreduce(vcat, setdiff(unique(dst_x.s),"")) do dst_s
    ii = dst_x.s .== dst_s
    os = map(setdiff(unique(sort(src_x.s)),"")) do s
      dst_x.s[ii] .= s
      (original = dst_s, replacement = s, f = sum(model(test_ds).data[oids,:]))
    end 
    dst_x.s[ii] .= dst_s
    os
  end |> DataFrame
end

function evaluatesubstitution(test_ds, dst_x::Matrix, src_x::Matrix, model, oids)
  size(dst_x, 1) > 1 && error("Not implemented due to Pevnak's Lazyness")
  mapreduce(vcat, unique(dst_x)) do dx
    ii = dst_x .== dx
    os = map(unique(sort(src_x[:]))) do sx
      dst_x[ii] .= sx
      (original = dx, replacement = sx, f = sum(model(test_ds).data[oids,:]))
    end 
    dst_x[ii] .= dx
    os
  end |> DataFrame
end

function evaluatesubstitution(test_ds, dst_x::Flux.OneHotMatrix, src_x::Flux.OneHotMatrix, model, oids)
  mapreduce(vcat, unique(dst_x.data)) do dx
    ii = findall(map(i -> dx.ix == i.ix, dst_x.data))
    os = map(unique(src_x.data)) do sx
      foreach(i -> dst_x.data[i] = sx, ii)
      (original = dx.ix, replacement = sx.ix, f = sum(model(test_ds).data[oids,:]))
    end 
    foreach(i -> dst_x.data[i] = dx, ii)
    os
  end |> DataFrame
end