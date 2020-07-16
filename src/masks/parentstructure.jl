using HierarchicalUtils

function idofnode(n, ns)
	ii = findall(map(x -> x === n, ns))
	isempty(ii) && return(0)
	only(ii)
end

function idofnode(n, ns::V) where {V<:Array{Pair{k,Int64} where k,1}}
	ii = findall(map(x -> x.first === n, ns))
	isempty(ii) && return(0)
	only(ii)
end

function allparents(nodes2parents, i::Int)
	i <= 0 && return()
	(i, allparents(nodes2parents, nodes2parents[i].second)...)
end

function allparents(parents_of_interest, nodes2parents, i::Int)
	i == 0 && return()
	id = idofnode(nodes2parents[i].first, parents_of_interest)
	parents = allparents(parents_of_interest, nodes2parents, nodes2parents[i].second)
	id == 0 ? parents : (id, parents...)
end

function firstparent(parents_of_interest, nodes2parents, i::Int)
	i == 0 && return(0)
	id = idofnode(nodes2parents[i].first, parents_of_interest)
	id == 0 ? firstparent(parents_of_interest, nodes2parents, nodes2parents[i].second) : id
end

function firstparents(parents_of_interest, nodes2parents)
	map(i -> parents_of_interest[i] => firstparent(parents_of_interest, nodes2parents, i), 1:length(parents_of_interest))
end

function parent_structure(ms)
	nodes = collect(NodeIterator(ms))
	nodes = filter(n -> !isa(n,ExplainMill.EmptyMask), nodes)
	nodes = unique(nodes)
	cs = zeros(Int, length(nodes))
	foreach(enumerate(nodes)) do (i,n)
		for c in children(n)
			ii = findall(map(x -> c === x, nodes))
			isempty(ii) && continue
			cs[only(ii)] = i
		end
	end;
	nodes2parents = map(x -> Pair(x...), zip(nodes, cs))
end
