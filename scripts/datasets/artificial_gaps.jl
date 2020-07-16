# for ((i=1;i<=20;i+=1)); do  for d in  one_of_1_2trees  one_of_1_5trees  one_of_1_paths  one_of_2_5trees  one_of_2_paths  one_of_5_paths ; do  julia -p 24 artificial.jl --dataset $d --incarnation $i ; done ; done
using Pkg
Pkg.activate("/home/tomas.pevny/julia/Pkg/ExplainMill.jl/scripts")

using ArgParse
using Flux
using Mill
using JsonGrinder
using JSON
using BSON
using Statistics
using IterTools
using TrainTools
using StatsBase
using ExplainMill
using Serialization
using Setfield
using DataFrames
using ExplainMill: jsondiff, nnodes, nleaves
include("common.jl")
include("loader.jl")
include("stats.jl")

_s = ArgParseSettings()
@add_arg_table! _s begin
  ("--dataset"; default="mutagenesis";arg_type=String);
  ("--task"; default="one_of_1_2trees";arg_type=String);
  ("--incarnation"; default=2;arg_type=Int);
  ("-k"; default=20;arg_type=Int);
end
settings = parse_args(ARGS, _s; as_symbols=true)
settings = NamedTuple{Tuple(keys(settings))}(values(settings))

###############################################################
# start by loading all samples
###############################################################
samples, labels, concepts = loaddata(settings);
resultsdir(s...) = simdir(settings.dataset, settings.task, "$(settings.incarnation)", s...)

###############################################################
# create schema of the JSON
###############################################################
if !isfile(resultsdir("model.jls"))
	!isdir(resultsdir()) && mkpath(resultsdir())
	sch = JsonGrinder.schema(samples);
	extractor = suggestextractor(sch);

	trndata = extractbatch(extractor, samples)
	function makebatch()
		i = rand(1:nobs(trndata), 100)
		trndata[i], Flux.onehotbatch(labels[i], 1:2)
	end
	ds = extractor(JsonGrinder.sample_synthetic(sch))
	good_model, concept_gap = nothing, 0
	for i in 1:10
		global good_model, concept_gap
		model = reflectinmodel(ds, d -> Dense(d,settings.k, relu), d -> SegmentedMeanMax(d), b = Dict("" => d -> Chain(Dense(d, settings.k, relu), Dense(settings.k, 2))));

		###############################################################
		#  train
		###############################################################
		opt = ADAM()
		ps = params(model)
		loss = (x,y) -> Flux.logitcrossentropy(model(x).data,y)

		cb = () -> begin
			o = model(trndata).data
			println("crossentropy = ",Flux.logitcrossentropy(o,Flux.onehotbatch(labels, 1:2)) ," accuracy = ",mean(Flux.onecold(softmax(o)) .== labels))
		end
		Flux.Optimise.train!(loss, ps, repeatedly(makebatch, 100000), opt, cb = Flux.throttle(cb, 60))
		print("trained: ");cb()
		soft_model = @set model.m.m = Chain(model.m.m...,  softmax);
		cg = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), 2), concepts))
		eg = ExplainMill.confidencegap(soft_model, extractor(JSON.parse("{}")), 1)
		@info "minimum gap on concepts = $(cg) on empty sample = $(eg)"
		if cg > 0 && eg > 0
			if cg > concept_gap
				good_model, concept_gap = model, cg 
			end 
		end 
		concept_gap > 0.95 && break
	end
	if concept_gap < 0 
		error("Failed to train a model")
	end
	model = good_model
	BSON.@save resultsdir("model.bson") model  extractor  schema 
end

d = BSON.load(resultsdir("model.bson"))
(model, extractor, sch) = d[:model], d[:extractor], d[:schema]
soft_model = @set model.m.m = Chain(model.m.m...,  softmax);
logsoft_model = @set model.m.m = Chain(model.m.m...,  logsoftmax);

###############################################################
#  Helper functions for explainability
###############################################################
const ci = TrainTools.classindexes(labels);

function loadclass(k, n = typemax(Int)) 
	dss =  map(extractor, sample(samples[ci[k]], min(n, length(ci[k])), replace = false))
	reduce(catobs, dss)
end

function onlycorrect(dss, i, min_confidence = 0)
	correct = ExplainMill.predict(soft_model, dss, [1,2]) .== i;
	dss = dss[correct[:]];
	min_confidence == 0 && return(dss)
	correct = ExplainMill.confidencegap(soft_model, dss, i) .>= min_confidence;
	dss[correct[:]]
end


###############################################################
#  Demonstration of explainability
###############################################################
# strain = "IP_PHONE", "GAME_CONSOLE", "SURVEILLANCE", "NAS", "HOME_AUTOMATION", "VOICE_ASSISTANT", "PC", "AUDIO", "MEDIA_BOX", "GENERIC_IOT", "IP_PHONE", "TV", "PRINTER", "MOBILE"
strain = 2
Random.seed!(settings.incarnation)
ds = loadclass(strain, 1000)
i = strain
concept_gap = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), i), concepts))
sample_gap = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), i), samples[labels .== 2]))
threshold_gap = floor(0.9*concept_gap, digits = 2) 
ds = onlycorrect(ds, strain, threshold_gap)
@info "minimum gap on concepts = $(concept_gap) on samples = $(sample_gap)"

exdf = isfile(resultsdir("stats.bson")) ? BSON.load(resultsdir("stats.bson"))[:exdf] : DataFrame()

explainers = [(ExplainMill.ConstExplainer(), "const")]
# for ((e, name), n, pruning_method, j) in Iterators.product(explainers, [0], [:oscilatingsfs, :sfs], 1:nobs(ds))
for ((e, name), n, pruning_method, j) in Iterators.product(explainers, [0], [:flatsfs, :flatsfsrr, :flatsfsos, :sfsrr], 1:nobs(ds))
	global exdf
	exdf = addexperiment(exdf, e, ds[j], logsoft_model, i, n, threshold_gap, name, pruning_method, j, settings)
end 
BSON.@save resultsdir("stats.bson") exdf


# explainers = [(ExplainMill.ConstExplainer(), "const"), (ExplainMill.StochasticExplainer(), "stochastic"), (ExplainMill.GradExplainer(), "grad"), (ExplainMill.GradExplainer2(), "grad2"), (ExplainMill.GradExplainer3(), "grad3")]
explainers = [(ExplainMill.StochasticExplainer(), "stochastic"), (ExplainMill.GradExplainer2(), "grad2")]
for ((e, name), n, pruning_method, j) in (Iterators.product(explainers, [0], [:greedy, :importantfirst, :oscilatingimportantfirst, :greedybreadthfirst, :breadthfirst2, :oscilatingbreadthfirst], 1:nobs(ds)))
	global exdf
	exdf = addexperiment(exdf, e, ds[j], logsoft_model, i, n, threshold_gap, name, pruning_method, j, settings)
end 
BSON.@save resultsdir("stats.bson") exdf


explainers = [(ExplainMill.GnnExplainer(), "gnn"), (ExplainMill.DafExplainer(true), "daf_prune"), (ExplainMill.DafExplainer(true, true), "banzhaf")]
# for ((e, name), n, pruning_method, j) in Iterators.product(explainers, [100, 200, 500], [:greedy, :importantfirst, :oscilatingimportantfirst, :breadthfirst2, :oscilatingbreadthfirst], 1:nobs(ds))
for ((e, name), n, pruning_method, j) in Iterators.product(explainers, [200], [:greedy, :importantfirst, :oscilatingimportantfirst, :greedybreadthfirst, :breadthfirst2, :oscilatingbreadthfirst], 1:nobs(ds))
	global exdf
	exdf = addexperiment(exdf, e, ds[j], logsoft_model, i, n, threshold_gap, name, pruning_method, j, settings)
end 

BSON.@save resultsdir("stats.bson") exdf