using DataFrames, Statistics, Serialization, PrettyTables, Printf, HypothesisTests

fixunderscore(x::Vector{String}) = replace.(x, "_" =>" ")
fixunderscore(x::Vector) = replace.(String.(x), "_" =>" ")
fixunderscore(x::String) = replace(x, "_" =>" ")
fixunderscore(x::Symbol) = fixunderscore(String(x))

fixtasks(x::String) = replace(x, "paths" => "1trees")

pruningdict = Dict(
	:sfsrr => "LbyL-GArr",
	:oscilatingsfs => "LbyL-GAos",
	:sfs => "LbyL-GAdd",
	:flatsfsrr => "GArr",
	:flatsfsos => "GAos",
	:flatsfs => "GAdd",
	:greedy => "HAdd",
	:importantfirst => "HArr",
	:oscilatingimportantfirst => "HAos",
	:breadthfirst2 => "LbyL-HArr",
	:greedybreadthfirst => "LbyL-HAdd",
	:oscilatingbreadthfirst => "LbyL-HAos",
	:abstemious => "ab",
	:importantlast => "il",
	)

function fixpruning(x::Symbol)
	if !haskey(pruningdict, x)
		@error "unknown $x"
		return(nothing)
	end
	return(pruningdict[x])
end

namesdict = Dict(
	"stochastic" => "Rnd",
 	"grad2" => "Grad",
 	"gnn" => "GNN", 
 	"gnn2" => "GNN2", 
 	"gnn3" => "GNN3", 
 	"pevnak" => "Bant",
 	"banzhaft" => "Bant",
 	"banzhaf" => "Banz"
	)

function fixnames(x)
	if !haskey(namesdict, x)
		@error "unknown $x"
		return(nothing)
	end
	return(namesdict[x])
end


function meanandconfidence(x)
	x = skipmissing(x)
	# ci = quantile(x, 0.975) - quantile(x, 0.025)
	# ci = quantile(x, 0.95) - quantile(x, 0.05)
	ci = confint(OneSampleTTest(Float64.(collect(x))))
	s = @sprintf("%.2f",ci[2] - ci[1])
	s = s[2:end]
	v = @sprintf("%.2f", mean(x))
	@sprintf("\$%s\\pm %s\$", v, s)
end

#####
#	Basic visualization
#####
uninformative = [:sfs, :oscilatingsfs, :sfsrr, :flatsfsrr, :flatsfsos, :flatsfs]
heuristic  = [:greedy, :importantfirst, :oscilatingimportantfirst, :breadthfirst2, :oscilatingbreadthfirst, :greedybreadthfirst]


flat_methods = [:flatsfsrr, :flatsfsos, :flatsfs, :greedy, :importantfirst, :oscilatingimportantfirst]

function filtercase(df, ranking::Nothing, level_by_level)
	pms = level_by_level ? [:sfs, :oscilatingsfs, :sfsrr] : [:flatsfs, :flatsfsrr, :flatsfsos]
	df₁ = filter(r -> r.name == "stochastic" && r.pruning_method ∈ pms, df)
	DataFrame(
	 ranking = "",
	 add_e = meanandconfidence(filter(r -> r.pruning_method == pms[1], df₁)[!,:excess_leaves]),
	 add_t = meanandconfidence(filter(r -> r.pruning_method == pms[1], df₁)[!,:time]),
	 addrr_e = meanandconfidence(filter(r -> r.pruning_method == pms[2], df₁)[!,:excess_leaves]),
	 addrr_t = meanandconfidence(filter(r -> r.pruning_method == pms[2], df₁)[!,:time]),
	 addrrft_e = meanandconfidence(filter(r -> r.pruning_method == pms[3], df₁)[!,:excess_leaves]),
	 addrrft_t = meanandconfidence(filter(r -> r.pruning_method == pms[3], df₁)[!,:time]),
	)
end

function filtercase(df, ranking::String, level_by_level)
	pms = level_by_level ? [:greedybreadthfirst, :breadthfirst2, :oscilatingbreadthfirst] : [:greedy, :importantfirst, :oscilatingimportantfirst,]
	df₁ = filter(r -> r.name == ranking && r.pruning_method ∈ pms, df)
	DataFrame(
	 ranking = fixnames(ranking),
	 add_e = meanandconfidence(filter(r -> r.pruning_method == pms[1], df₁)[!,:excess_leaves]),
	 add_t = meanandconfidence(filter(r -> r.pruning_method == pms[1], df₁)[!,:time]),
	 addrr_e = meanandconfidence(filter(r -> r.pruning_method == pms[2], df₁)[!,:excess_leaves]),
	 addrr_t = meanandconfidence(filter(r -> r.pruning_method == pms[2], df₁)[!,:time]),
	 addrrft_e = meanandconfidence(filter(r -> r.pruning_method == pms[3], df₁)[!,:excess_leaves]),
	 addrrft_t = meanandconfidence(filter(r -> r.pruning_method == pms[3], df₁)[!,:time]),
	)
end

df[!,:task] = fixtasks.(df[!,:task])
adf = mapreduce(vcat, [false, true]) do b
	mapreduce(vcat, [nothing,  "gnn", "gnn2", "grad2", "banzhaf", "banzhaft", "stochastic" ]) do r
		filtercase(df, r, b)
	end
end

# pretty_table(adf, backend = :latex, formatters = ft_printf("%.2f"))
pretty_table(adf, backend = :latex)
# pretty_table(adf)
