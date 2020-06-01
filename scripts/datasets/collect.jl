using Distributed
@everywhere using Pkg
@everywhere Pkg.activate("..")
@everywhere using BSON
@everywhere using DataFrames
@everywhere using Statistics
@everywhere using Serialization

@everywhere rdir = "/home/tomas.pevny/data/sims/"

@everywhere begin 
	function loadproblem(d)
		@show d
		ii = filter(i -> isfile(joinpath(rdir,d,i,"stats.bson")), readdir(joinpath(rdir,d)))
		dfs = map(ii) do i 
			try
				return(BSON.load(joinpath(rdir,d,i,"stats.bson"))[:exdf])
			except
				@error "failed in $(joinpath(d,i,"stats.bson"))"
				return(nothing)
			end
		end
		if !isempty(filter(s -> isa(s, Type), dfs))
			m = ii[isa.(dfs, Type)]
			@error "rerun $d $(m) as there is a type"
		end
		dfs = filter(s -> isa(s, DataFrame), dfs)
		dfs = filter(s -> size(s,1) > 0, dfs)
		vcat(dfs...)
	end
end

function checkmodel(d)
	ii = filter(i -> isfile(joinpath(rdir,d,i,"model.bson")), readdir(joinpath(rdir,d)))
	@show (d, length(ii))
end

os = map(readdir(rdir)) do problem
	os = pmap(readdir(joinpath(rdir, problem))) do task 
		loadproblem(joinpath(problem, task))
	end
	reduce(vcat, os)
end;
os = filter(!isempty, os)
df = reduce(vcat, os)
ns = setdiff(names(df), [:dataset, :task, :name, :pruning_method, :n]);
df = by(df, [:dataset, :task, :name, :pruning_method, :n], df -> DataFrame([k => mean(skipmissing(df[!,k])) for k in ns]...))
serialize("results.jls", df)
