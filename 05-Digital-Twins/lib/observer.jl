
using Configurations: @option
using Ensembles: Ensembles, AbstractOperator
using DrWatson: projectdir
using Random: Random
using Ensembles: Ensembles, NoisyObserver, KeyObserver, get_state_keys

export get_observer
function get_observer(options::NoisyObservationOptions)
    return NoisyObserverConfigurations(collect(options.keys); params=options)
end

function get_observer(options::Nothing)
    return NothingObserver()
end

export MultiTimeObserver
struct MultiTimeObserver{T}
    times_observers::Vector{Pair{T,Any}}
    times::Vector{T}
    observers::Vector
    unique_times::Vector{T}
end

function MultiTimeObserver(times_observers::Vector{Pair{Float64,Any}})
    sort!(times_observers; by=to -> to.first)
    times = [to.first for to in times_observers]
    observers = [to.second for to in times_observers]
    unique_times = unique(times)
    return MultiTimeObserver(times_observers, times, observers, unique_times)
end

export get_multi_time_observer
function get_multi_time_observer(options::MultiTimeObserverOptions)
    return MultiTimeObserver(collect(options.observers))
end

struct NothingObserver <: AbstractOperator end

Ensembles.xor_seed!(::NothingObserver, seed_mod::UInt) = nothing
(M::NothingObserver)(member::Dict{Symbol,Any}) = Dict{Symbol,Any}()
function Ensembles.split_clean_noisy(::NothingObserver, obs::Dict{Symbol,<:Any})
    return (Dict{Symbol,Any}(), Dict{Symbol,Any}())
end

function NoisyObserverConfigurations(op::Ensembles.AbstractOperator; params)
    noise_scale = params.noise_scale
    seed = params.seed
    rng = Random.MersenneTwister(seed)
    if seed == 0
        seed = Random.rand(UInt64)
    end
    Random.seed!(rng, seed)
    state_keys = get_state_keys(op)
    if !params.only_noisy
        state_keys = append!(
            [Symbol(key, :_noisy) for key in get_state_keys(op)], state_keys
        )
    end

    return NoisyObserver(op, state_keys, noise_scale, rng, seed, params.only_noisy)
end

function NoisyObserverConfigurations(state_keys; params)
    return NoisyObserverConfigurations(KeyObserver(state_keys); params)
end
