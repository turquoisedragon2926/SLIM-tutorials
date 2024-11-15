
params_file = abspath(ARGS[1])
include("install.jl")

using TerminalLoggers: TerminalLogger
using Logging: global_logger
isinteractive() && global_logger(TerminalLogger())
using ProgressLogging: @withprogress, @logprogress

using CairoMakie: CairoMakie
using Makie: with_theme, theme_latexfonts, update_theme!

using DrWatson: srcdir, datadir, plotsdir, produce_or_load, wsave, projectdir, scriptsdir
using Format: cfmt
using JutulJUDIFilter
using Statistics: mean, std

using FilterComparison

include(scriptsdir("generate_initial_ensemble.jl"))
include(scriptsdir("run_estimator.jl"))

# Read data.
params = include(params_file)
data_ensemble, _, filestem_ensemble = produce_or_load_run_estimator(
    params; loadfile=true, force=false
)

state_means = data_ensemble["state_means"]
state_times = data_ensemble["state_times"]
observation_means = data_ensemble["observation_means"]
observation_clean_means = data_ensemble["observation_clean_means"]
observation_times = data_ensemble["observation_times"]
observations_clean = data_ensemble["observations_clean"]
observations = data_ensemble["observations"]
states = data_ensemble["states"]
logs = data_ensemble["logs"]

save_dir_root = plotsdir("estimator_ensemble", "states", filestem_ensemble)
with_theme(theme_latexfonts()) do
    update_theme!(; fontsize=24)
    CairoMakie.activate!()

    state_keys = collect(keys(state_means[1]))

    plot_states(
        state_times,
        state_means,
        params.estimator;
        save_dir_root=joinpath(save_dir_root, "mean"),
    )

    state_stds = [std(ensemble; state_keys=state_keys) for ensemble in states]
    plot_states(
        state_times,
        state_stds,
        params.estimator;
        save_dir_root=joinpath(save_dir_root, "std"),
    )

    for i in 1:min(length(states[1].members), 2)
        states_member_i = [e.members[i] for e in states]
        plot_states(
            state_times,
            states_member_i,
            params.estimator;
            save_dir_root=joinpath(save_dir_root, "e$i"),
            try_interactive=false,
        )
    end

    for (i, ensemble) in enumerate(states)
        save_dir = joinpath(save_dir_root, "t$i")
        plot_states(
            1:length(ensemble.members),
            ensemble.members,
            params.estimator;
            save_dir_root=save_dir,
        )
    end
end

nothing
