using Distributed
@everywhere using Pkg
@everywhere Pkg.activate("..")
@everywhere using BSON
@everywhere using DataFrames
@everywhere using Statistics
@everywhere using Serialization

@everywhere rdir = "/home/tomas.pevny/data/sims/"

@everywhere begin 
	function loadproblem(d, i)
		# files = filter(s -> startswith(s, "stats4_"), readdir(joinpath(rdir,d,i)))
			files = filter(s -> startswith(s, "gnn_"), readdir(joinpath(rdir,d,i)))
		isempty(files) && return(DataFrame())
		os = mapreduce(vcat, files) do f 
			try 
				BSON.load(joinpath(rdir,d,i,f))[:exdf]
			catch
				println("failed on $(joinpath(rdir,d,i,f))")
				DataFrame()
			end
		end
	end

	function loadproblem(d)
		ii = readdir(joinpath(rdir,d))
		dfs = map(i -> loadproblem(d, i), ii)
		dfs = filter(s -> size(s,1) > 0, dfs)
		vcat(dfs...)
	end
end

function checkmodel(d)
	ii = filter(i -> isfile(joinpath(rdir,d,i,"model.bson")), readdir(joinpath(rdir,d)))
	@show (d, length(ii))
end

function checkcompletness!(miss, df, combinations, name, pr)
	for c in eachrow(combinations)
		o = filter(r -> r.dataset == c.dataset
				&& r.task == c.task 
				&& r.incarnation == c.incarnation 
				&& r.name == name 
				&& r.pruning_method == pr, df)
		if isempty(o) 
			println((name, pr, c.dataset, c.task, c.incarnation))
			push!(miss, (name, pr, c.dataset, c.task, c.incarnation))
		end
	end
end

function checkcompletness(df)
	heuristic  = [:greedy, :importantfirst, :oscilatingimportantfirst, :greedybreadthfirst, :breadthfirst2, :oscilatingbreadthfirst]
	uninformative = [:flatsfs, :flatsfsrr, :flatsfsos, :sfs, :sfsrr, :oscilatingsfs]
	combinations = by(df, [:dataset, :task, :incarnation], df -> ())
	miss = []
	for (name, pr) in Iterators.product(["stochastic"], vcat(heuristic, uninformative))
		subdf = filter(r -> r.name == name && r.pruning_method == pr, df)
		checkcompletness!(miss, subdf, combinations, name, pr)
	end

	for (name, pr) in Iterators.product(["grad2", "banzhaf", "pevnak", "gnn"], heuristic)
		subdf = filter(r -> r.name == name && r.pruning_method == pr, df)
		checkcompletness!(miss, subdf, combinations, name, pr)
	end
	miss
end

os = map(["deviceid", "hepatitis", "mutagenesis"]) do problem
	os = pmap(readdir(joinpath(rdir, problem))) do task 
		loadproblem(joinpath(problem, task))
	end
	os = filter(!isempty, os)
	isempty(os) && return(DataFrame())
	reduce(vcat, os)
end;
size.(os)
os = filter(!isempty, os)
df = reduce(vcat, os)

heuristic  = [:greedy, :importantfirst, :oscilatingimportantfirst, :greedybreadthfirst, :breadthfirst2, :oscilatingbreadthfirst]
uninformative = [:flatsfs, :flatsfsrr, :flatsfsos, :sfs, :sfsrr, :oscilatingsfs]
dff = filter(r -> r.name ∈ ["Rnd", "Grad", "GNN", "Banz", "GNN2"], df)
dff = filter(r -> r.pruning_method ∈ vcat(heuristic, uninformative), dff)

