if abspath(PROGRAM_FILE) == @__FILE__
    params_file = abspath(ARGS[1])
end

include("install.jl")

using TerminalLoggers: TerminalLogger
using Logging: global_logger
global_logger(TerminalLogger())
using ProgressLogging: @withprogress, @logprogress

using DrWatson: wsave, datadir, produce_or_load, srcdir
using Ensembles:
    Ensembles, get_state_keys, get_ensemble_matrix, split_clean_noisy, xor_seed!
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
include(srcdir("seismic_co2_model.jl"))

# Generate synthetic ground-truth observations.
function generate_ground_truth(params)
    K = (Val(:Saturation), Val(:Pressure), Val(:Permeability))
    # K = (Val(:Saturation), Val(:Pressure))
    # K = (Val(:OverallMoleFraction), Val(:Pressure))
    JMT = JutulModelTranslator(K)

    ## Set seed for ground-truth simulation.
    Random.seed!(0xabceabd47cada8f4)

    M = JutulModel(; translator=JMT, options=params.transition, kwargs=(;info_level=-1))

    ## Make operators.
    observers = get_multi_time_observer(params.observation)
    max_transition_step = params.max_transition_step

    ground_truth = @time let
        state = Dict{Symbol,Any}()
        initialize_member!(M, state)

        ## Generate states and observations.
        t0 = 0.0
        state_times = Float64[]
        states = Dict{Symbol,Any}[]
        observations = Vector{Dict{Symbol,Any}}(undef, length(observers.times))
        observations_clean = Vector{Dict{Symbol,Any}}(undef, length(observers.times))
        push!(states, deepcopy(state))
        push!(state_times, 0.0)
        obs_idx = 1
        tf = observers.times[end]

        @withprogress name = "Ground-truth" begin
            cache = Dict()
            @logprogress t0 / tf
            for (t, observer_options) in observers.times_observers
                if t0 != t
                    if !isnothing(max_transition_step)
                        while t0 + max_transition_step < t
                            state = M(state, t0, t0 + max_transition_step)
                            t0 += max_transition_step
                            push!(states, deepcopy(state))
                            push!(state_times, t0)
                            @logprogress t0 / tf
                        end
                    end
                    state = M(state, t0, t)
                    push!(states, deepcopy(state))
                    push!(state_times, t)
                end
                Random.seed!(0xabceabd47cada8f4 ⊻ hash(t))
                observer = if haskey(cache, observer_options)
                    println("Cache success at time $(t)!")
                    cache[observer_options]
                else
                    get_observer(observer_options)
                end
                xor_seed!(observer, UInt64(0xabc2fe2e546a031c) ⊻ hash(t))
                obs = observer(state)
                observations_clean[obs_idx], observations[obs_idx] = split_clean_noisy(
                    observer, obs
                )
                obs_idx += 1
                t0 = t
                @logprogress t0 / tf
            end
        end
        (;
            states,
            observations,
            observations_clean,
            state_times,
            observation_times=observers.times,
        )
    end
    println("  ^ timing for making ground truth data")
    return data = Dict(
        "states" => ground_truth.states,
        "observations" => ground_truth.observations,
        "observations_clean" => ground_truth.observations_clean,
        "state_times" => ground_truth.state_times,
        "observation_times" => ground_truth.observation_times,
    )
end

function ground_truth_stem(params)
    return string(hash(params.ground_truth); base=62)
end

function produce_or_load_ground_truth(params::JutulJUDIFilterOptions; filestem=nothing, kwargs...)
    params_gt = params.ground_truth
    if isnothing(filestem)
        filestem = ground_truth_stem(params)
    end

    params_file = datadir("ground_truth", "params", "$filestem.jld2")
    wsave(params_file; params=params_gt)

    params_file = datadir("ground_truth", "params", "$filestem-human.yaml")
    YAML.write_file(params_file, to_dict(params_gt, YAMLStyle))

    savedir = datadir("ground_truth", "data")
    data, filepath = produce_or_load(
        generate_ground_truth,
        params_gt,
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
    params_file = abspath(ARGS[1])
    params = include(params_file)
    produce_or_load_ground_truth(params)
end
