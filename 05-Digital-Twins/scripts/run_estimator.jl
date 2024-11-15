if abspath(PROGRAM_FILE) == @__FILE__
    params_file = abspath(ARGS[1])
end

include("install.jl")

using TerminalLoggers: TerminalLogger
using Logging: global_logger
using ProgressLogging: @progress
global_logger(TerminalLogger())

using DrWatson: wsave, datadir, produce_or_load, srcdir, projectdir, scriptsdir
using Ensembles:
    Ensembles,
    Ensemble,
    get_state_keys,
    get_ensemble_matrix,
    split_clean_noisy,
    xor_seed!,
    get_ensemble_members
using Random: Random

using ConfigurationsJutulDarcy
using Configurations: to_dict, YAMLStyle
using JutulDarcy
using JutulDarcy.Jutul
using Statistics
using LinearAlgebra
using YAML: YAML

using ImageTransformations: ImageTransformations
using JLD2: JLD2

using FilterComparison

include(srcdir("jutul_model.jl"))
include(srcdir("estimator.jl"))
include(srcdir("filter_loop.jl"))

include(scriptsdir("generate_ground_truth.jl"))
include(scriptsdir("generate_initial_ensemble.jl"))

function run_estimator(params)
    params_estimator = params.estimator
    data_gt, _ = produce_or_load_ground_truth(params; loadfile=true)

    data_initial, _ = produce_or_load_initial_ensemble(params; loadfile=true)

    states_gt = data_gt["states"]
    observations_gt = data_gt["observations"]

    ensemble = data_initial["ensemble"]

    K = (Val(:Saturation), Val(:Pressure), Val(:Permeability))
    JMT = JutulModelTranslator(K)

    M = JutulModel(; translator=JMT, options=params_estimator.transition, kwargs=(;info_level=-1))
    observers = get_multi_time_observer(params_estimator.observation)

    # Initialize member for all primary variables in simulation.
    @progress "Initialize ensemble states" for member in get_ensemble_members(ensemble)
        initialize_member!(M, member)
    end

    estimator = get_estimator(params_estimator.algorithm)

    empty!(ensemble.state_keys)
    append!(ensemble.state_keys, params_estimator.assimilation_state_keys)

    t0 = 0.0
    return data = filter_loop(
        ensemble,
        t0,
        estimator,
        M,
        observers,
        observations_gt;
        name=get_short_name(params_estimator.algorithm),
        max_transition_step=params_estimator.max_transition_step,
        assimilation_obs_keys=params_estimator.assimilation_obs_keys,
    )
end

function filter_stem(params)
    return ground_truth_stem(params) *
           "-" *
           initial_ensemble_stem(params) *
           "-" *
           string(hash(params.estimator); base=62)
end

function produce_or_load_run_estimator(params; filestem=nothing, kwargs...)
    params_estimator = params.estimator
    if isnothing(filestem)
        filestem = filter_stem(params)
    end

    params_file = datadir("estimator", "params", "$filestem.jld2")
    wsave(params_file; params=params_estimator)

    params_file = datadir("estimator", "params", "$filestem-human.yaml")
    YAML.write_file(params_file, to_dict(params_estimator, YAMLStyle))

    savedir = datadir("estimator", "data")
    data, filepath = produce_or_load(
        run_estimator,
        params,
        savedir;
        filename=filestem,
        verbose=true,
        tag=false,
        loadfile=false,
        kwargs...,
    )
    return data, filepath, filestem
end

if abspath(PROGRAM_FILE) == @__FILE__
    params = include(params_file)
    produce_or_load_run_estimator(params)
end
