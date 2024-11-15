
using Random: Random
using ProgressLogging: @withprogress, @logprogress
using Ensembles: assimilate_data, split_clean_noisy

using FilterComparison

function filter_loop(
    ensemble,
    t0,
    estimator,
    transitioner,
    observers,
    observations_gt;
    name="Time",
    max_transition_step=nothing,
    assimilation_obs_keys=nothing,
)
    logs = []
    states = []
    observations = []
    observations_clean = []
    cache = Dict()
    name_orig = name
    state_times = []
    observation_times = []
    state_means = []
    observation_means = []
    observation_clean_means = []
    progress_name = name * ": "
    state_keys = collect(keys(ensemble.members[1]))
    @time begin
        push!(states, deepcopy(ensemble))
        push!(state_means, mean(ensemble; state_keys=state_keys))
        push!(state_times, t0)
        tf = observers.times[end]
        @withprogress name = progress_name begin
            for ((t, observer_options), y_obs) in
                zip(observers.times_observers, observations_gt)
                ## Advance ensemble to time t.
                if t0 != t
                    if !isnothing(max_transition_step)
                        while t0 + max_transition_step < t
                            ensemble = transitioner(
                                ensemble, t0, t0 + max_transition_step; inplace=true
                            )
                            t0 += max_transition_step
                            @logprogress t0 / tf
                            push!(states, deepcopy(ensemble))
                            push!(state_means, mean(ensemble; state_keys=state_keys))
                            push!(state_times, t0)
                        end
                    end
                    ensemble = transitioner(ensemble, t0, t; inplace=true)
                    push!(states, deepcopy(ensemble))
                    push!(state_means, mean(ensemble; state_keys=state_keys))
                    push!(state_times, t)
                end

                Random.seed!(0xabceabd47cada8f4 ⊻ hash(t))
                observer = if haskey(cache, observer_options)
                    cache[observer_options]
                else
                    get_observer(observer_options)
                end
                xor_seed!(observer, UInt64(0xabc2fe2e546a031c) ⊻ hash(t))

                ## Take observation at time t.
                ensemble_obs = observer(ensemble)
                ensemble_obs_clean, ensemble_obs_noisy = split_clean_noisy(
                    observer, ensemble_obs
                )

                if !isnothing(assimilation_obs_keys)
                    empty!(ensemble_obs_clean.state_keys)
                    append!(ensemble_obs_clean.state_keys, assimilation_obs_keys)

                    empty!(ensemble_obs_noisy.state_keys)
                    append!(ensemble_obs_noisy.state_keys, assimilation_obs_keys)
                end

                ## Record.
                push!(observation_means, mean(ensemble_obs))
                push!(observation_clean_means, mean(ensemble_obs_clean))
                push!(observation_times, t)
                push!(observations_clean, deepcopy(ensemble_obs_clean))
                push!(observations, deepcopy(ensemble_obs_noisy))

                if !isnothing(estimator)
                    ## Assimilate observation
                    log_data = Dict{Symbol,Any}()
                    (ensemble, timing...) = @timed assimilate_data(
                        estimator,
                        ensemble,
                        ensemble_obs_clean,
                        ensemble_obs_noisy,
                        y_obs,
                        log_data,
                    )
                    log_data[:timing] = timing

                    ## Record.
                    push!(logs, log_data)
                    push!(states, deepcopy(ensemble))
                    push!(state_means, mean(ensemble; state_keys=state_keys))
                    push!(state_times, t)
                end
                @logprogress t0 / tf
            end
        end
    end
    println("  ^ timing for running filter loop ($name_orig)")

    data = Dict(
        "states" => states,
        "state_means" => state_means,
        "state_times" => state_times,
        "observation_means" => observation_means,
        "observation_clean_means" => observation_clean_means,
        "observation_times" => observation_times,
        "observations_clean" => observations_clean,
        "observations" => observations,
        "logs" => logs,
        "ensemble" => states[end],
        "t" => state_times[end],
    )
    return data
end
