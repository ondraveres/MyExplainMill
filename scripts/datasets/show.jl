using DataFrames, Statistics, Serialization, PGFPlotsX

fixunderscore(x::Vector{String}) = replace.(x, "_" =>" ")
fixunderscore(x::Vector) = replace.(String.(x), "_" =>" ")
fixunderscore(x::String) = replace(x, "_" =>" ")
fixunderscore(x::Symbol) = fixunderscore(String(x))

fixtasks(x::String) = replace(x, "paths" => "1trees")

namesdict = Dict(
	:breadthfirst2 => "bf",
	:greedy => "gr",
	:abstemious => "ab",
	:importantlast => "il",
	:importantfirst => "if",
	:oscilatingimportantfirst => "oif",
	:oscilatingbreadthfirst => "obf",
	:oscilatingsfs => "ofs",
	:sfs => "fs",
	)

function fixnames(x::Symbol)
	if !haskey(namesdict, x)
		@error "unknown $x"
		return(nothing)
	end
	return(namesdict[x])
end

df = deserialize("results.jls")
df[!,:task] = fixtasks.(df[!,:task])
ns = setdiff(names(df), [:dataset, :task, :name, :pruning_method, :n])
#####
#	Basic visualization
#####
# df₁ = filter(row -> row[:name] ∈ ["const","stochastic", "grad", "grad2", "grad3"], df)
df₁ = filter(row -> row[:name] ∈ ["stochastic", "grad2"], df)
df₃ = filter(row -> row[:name] ∈ ["const"] && row.pruning_method == :sfs , df)
df₃[!,:name] .= "stochastic"
# df₂ = filter(row -> row[:pruning_method] ∈ [:greedy, :importantfirst, :breadthfirst2, :breadthfirst] && (row[:n] == 200), df)
df₂ = filter(row -> row[:n] == 200, df)
dff = vcat(df₁, df₂, df₃)
# dff = filter(row -> row[:name] != "daf_layerwise", dff)
dff = filter(row -> row.pruning_method ∉ [:breadthfirst, :abstemious, :importantlast] , dff)
dff = by(dff, [:name, :pruning_method], df -> DataFrame([k => mean(skipmissing(df[!,k])) for k in ns]...))

for (k, title, ymax, ofname) in [
			(:excess_leaves, "Excessive terms in explanation", 0.7, "excess_leaves"), 
			(:excess_leaves, "Excessive terms in explanation", 30, "excess_leaves_full"), 
			(:time, "{time of explanation}", 2, "time"), 
			(:misses_leaves, "Missing terms in explanation" ,2 , "misess_leaves")
			]
	# global dff
	p = @pgf Axis(
		{	ybar,
			ymax = ymax,
			ymin = 0,
			"legend style"={
				at={"(0.5,-0.2)"},
				anchor="north",
				"legend columns"=-1,
				}, 
			ylabel= fixunderscore(k),
			"symbolic x coords"= fixnames.(sort(unique(dff[!,:pruning_method]))), 
			xtick="data",
		    # title = title,
			"x tick label style"={rotate=45,anchor="east"}, 
			"bar width"="3pt",
	    },
	    );

	for m in ["gnn", "banzhaf", "grad2", "stochastic"]
		dt = filter(row -> row[:name] == m, dff)
		dt = by(dt, [:pruning_method], df -> DataFrame([k => mean(skipmissing(df[!,k])) for k in ns]...))
		push!(p, Plot(Coordinates(fixnames.((dt[!,:pruning_method])), dt[!,k])))
	end
	# labels = fixunderscore(sort(unique(String.(dff[!,:name]))))
	labels = ["gnn", "banzhaf", "grad2", "stochastic"]
	push!(p, Legend(labels))
	PGFPlotsX.save("$(ofname).pdf",p)
end


df₁ = filter(row -> row[:name] ∈ ["stochastic", "grad2"], df)
df₃ = filter(row -> row[:name] ∈ ["const"] && row.pruning_method == :sfs , df)
df₃[!,:name] .= "stochastic"
df₂ = filter(row -> row[:n] == 200, df)
dff = vcat(df₁, df₂, df₃)
dff = filter(row -> row.pruning_method ∉ [:breadthfirst, :abstemious, :importantlast] , dff)
dss = sort(unique(dff[!,:dataset]))
tasks = sort(unique(dff[!,:task]))

k = :excess_leaves
# k = :time
title = "Excessive terms in explanation"
@pgf gp = GroupPlot({group_style = { group_size = "$(length(dss)) by $(length(tasks))"}});
for (j,ts) in enumerate(tasks)
	for (i,ds) in enumerate(dss)
		dtt = filter(row -> row.task == ts && row.dataset == ds, dff)
		if isempty(dtt)
			push!(gp, @pgf nothing)
			continue
		end
		t = j == 1 ? ds : ""
		ylab = i == 1 ? fixunderscore(ts) : ""
		p = @pgf Axis(
		{	ybar,
			enlargelimits=0.25,
			# ymax = 2,
			ymin = 0,
			title = t,
			ylabel = ylab,
			"legend style"={
				at={"(0.5,-0.2)"},
				anchor="north",
				"legend columns"=-1,
				}, 
			"symbolic x coords"= fixnames.(sort(unique(dtt[!,:pruning_method]))), 
			xtick="data",
			"x tick label style"={rotate=45,anchor="east"}, 
			"bar width"="2pt",
	    },
	    );

		for m in sort(unique(dtt[!,:name]))
			dt = filter(row -> row[:name] == m, dtt)
			push!(p, Plot(Coordinates(fixnames.(dt[!,:pruning_method]), dt[!,k])))
		end
		labels = fixunderscore(sort(unique(String.(dtt[!,:name]))))
		(j == length(tasks) && i == length(dss)) && push!(p, Legend(labels))
		push!(gp, p)
	end	
end
gp
# PGFPlotsX.save("group_excess.pdf",gp)
PGFPlotsX.save("group_time.pdf",gp)