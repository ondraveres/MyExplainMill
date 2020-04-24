using HierarchicalUtils

function find_influential_parent(cs, nodes, i)
	cs[i] == 0 && return(0)
	isa(nodes[cs[i]], ExplainMill.AbstractNoMask) && return(find_influential_parent(cs, nodes, cs[i]))
	return(cs[i])
end

function parent_structure(mask)
	nodes = collect(NodeIterator(mask))
	nodes = filter(n -> !isa(n,ExplainMill.EmptyMask), nodes)
	cs = zeros(Int, length(nodes))
	foreach(enumerate(nodes)) do (i,n)
		for c in children(n)
			ii = findall(map(x -> c == x, nodes))
			isempty(ii) && continue
			cs[only(ii)] = i
		end
	end;
	map(x -> Pair(x...), zip(nodes, cs))
end

function remove_useless_parents(vs)
	remove_useless_parents([v.first for v in vs],[v.second for v in vs])
end

function remove_useless_parents(nodes, cs)
	csi = map(i -> find_influential_parent(cs, nodes, i), 1:length(cs))
	useless = map(n -> isa(n, ExplainMill.AbstractNoMask), nodes)
	csi = map(i -> i - sum(useless[1:i]), csi)
	map(x -> Pair(x...), zip(nodes[.!useless], csi[.!useless]))
end

