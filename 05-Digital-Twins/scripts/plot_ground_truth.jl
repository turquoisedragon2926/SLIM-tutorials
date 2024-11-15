
params_file = abspath(ARGS[1])
include("install.jl")

using TerminalLoggers: TerminalLogger
using Logging: global_logger
isinteractive() && global_logger(TerminalLogger())

using CairoMakie: CairoMakie
using Makie: with_theme, theme_latexfonts, update_theme!

using DrWatson: srcdir, datadir, plotsdir, produce_or_load, wsave, scriptsdir, projectdir
using JutulJUDIFilter

using FilterComparison
include(scriptsdir("generate_ground_truth.jl"))
include(srcdir("seismic_utils.jl"))

# Read data.
params = include(params_file)
data_gt, _, filestem_gt = produce_or_load_ground_truth(params; loadfile=true, force=false)

states = data_gt["states"]
observations = data_gt["observations"]
observations_clean = data_gt["observations_clean"]
state_times = data_gt["state_times"]
observation_times = data_gt["observation_times"]

with_theme(theme_latexfonts()) do
    update_theme!(; fontsize=24)
    CairoMakie.activate!()

    save_dir_root = plotsdir("ground_truth", filestem_gt, "static")
    if isa(params.ground_truth.observation.observers[1].second, SeismicCO2ObserverOptions)
        params_seismic = params.ground_truth.observation.observers[1].second.seismic
        (; velocity, density, velocity0, density0) = read_static_seismic_params(
            params_seismic
        )

        n = params_seismic.mesh.n
        d = params_seismic.mesh.d
        idx_wb = maximum(find_water_bottom_immutable(log.(velocity) .- log(velocity[1, 1])))
        idx_unconformity = find_water_bottom_immutable(
            (velocity .- 3500.0f0) .* (velocity .≥ 3500.0f0)
        )
        src_positions, rec_positions = build_source_receiver_geometry(
            n, d, idx_wb; params=params_seismic.source_receiver_geometry
        )
        plot_points_of_interest(
            params.ground_truth,
            src_positions,
            rec_positions;
            idx_wb,
            idx_unconformity,
            save_dir_root,
            try_interactive=false,
        )

        plot_states(
            [0],
            [velocity],
            params.ground_truth,
            Val(:velocity);
            save_dir_root,
            try_interactive=false,
        )

        plot_states(
            [0],
            [density],
            params.ground_truth,
            Val(:density);
            save_dir_root,
            try_interactive=false,
        )
        plot_states(
            [0],
            [velocity0],
            params.ground_truth,
            Val(:velocity0);
            save_dir_root,
            try_interactive=false,
        )
        plot_states(
            [0],
            [density0],
            params.ground_truth,
            Val(:density0);
            save_dir_root,
            try_interactive=false,
        )
        if haskey(states[1], :Permeability)
            plot_states(
                state_times,
                states,
                params.ground_truth,
                Val(:Permeability);
                save_dir_root,
                try_interactive=false,
            )
        end
    end

    save_dir_root = plotsdir("ground_truth", filestem_gt, "states")
    if haskey(states[1], :Saturation)
        plot_states(
            state_times,
            states,
            params.ground_truth,
            Val(:Saturation);
            save_dir_root,
            try_interactive=false,
        )
    end

    if haskey(states[1], :Saturation) && haskey(states[1], :Permeability)
        plot_states(
            state_times,
            states,
            params.ground_truth,
            Val(:Saturation_Permeability);
            save_dir_root,
            try_interactive=false,
        )
    end

    if haskey(states[1], :Pressure)
        plot_states(
            state_times,
            states,
            params.ground_truth,
            Val(:Pressure);
            save_dir_root,
            try_interactive=false,
        )
        plot_states(
            state_times,
            states,
            params.ground_truth,
            Val(:Pressure_diff);
            save_dir_root,
            try_interactive=false,
        )
    end

    save_dir_root = plotsdir("ground_truth", filestem_gt, "observations")
    if haskey(observations[1], :rtm)
        plot_states(
            observation_times,
            observations,
            params.ground_truth,
            Val(:rtm);
            save_dir_root,
            try_interactive=false,
        )
        plot_states(
            observation_times,
            observations,
            params.ground_truth,
            Val(:rtm_diff);
            save_dir_root,
            try_interactive=false,
        )
    end

    if haskey(observations[1], :dshot)
        plot_states(
            observation_times,
            observations,
            params.ground_truth,
            Val(:dshot);
            save_dir_root,
            try_interactive=false,
        )
        plot_states(
            observation_times,
            observations,
            params.ground_truth,
            Val(:dshot_diff);
            save_dir_root,
            try_interactive=false,
        )
    end

    save_dir_root = plotsdir("ground_truth", filestem_gt, "observations_clean")
    if haskey(observations_clean[1], :rtm)
        plot_states(
            observation_times,
            observations_clean,
            params.ground_truth,
            Val(:rtm);
            save_dir_root,
            try_interactive=false,
        )
        plot_states(
            observation_times,
            observations_clean,
            params.ground_truth,
            Val(:rtm_diff);
            save_dir_root,
            try_interactive=false,
        )
    end

    if haskey(observations_clean[1], :dshot)
        plot_states(
            observation_times,
            observations_clean,
            params.ground_truth,
            Val(:dshot);
            save_dir_root,
            try_interactive=false,
        )
        plot_states(
            observation_times,
            observations_clean,
            params.ground_truth,
            Val(:dshot_diff);
            save_dir_root,
            try_interactive=false,
        )
    end

    if haskey(observations_clean[1], :density)
        plot_states(
            observation_times,
            observations_clean,
            params.ground_truth,
            Val(:density);
            save_dir_root,
            try_interactive=false,
        )
        plot_states(
            observation_times,
            observations_clean,
            params.ground_truth,
            Val(:density_diff);
            save_dir_root,
            try_interactive=false,
        )
    end

    if haskey(observations_clean[1], :velocity)
        plot_states(
            observation_times,
            observations_clean,
            params.ground_truth,
            Val(:velocity);
            save_dir_root,
            try_interactive=false,
        )
        plot_states(
            observation_times,
            observations_clean,
            params.ground_truth,
            Val(:velocity_diff);
            save_dir_root,
            try_interactive=false,
        )
    end

    if haskey(observations_clean[1], :density) && haskey(observations_clean[1], :velocity)
        plot_states(
            observation_times,
            observations_clean,
            params.ground_truth,
            Val(:impedance);
            save_dir_root,
            try_interactive=false,
        )
        plot_states(
            observation_times,
            observations_clean,
            params.ground_truth,
            Val(:impedance_diff);
            save_dir_root,
            try_interactive=false,
        )
        plot_states(
            observation_times,
            observations_clean,
            params.ground_truth,
            Val(:impedance_reldiff);
            save_dir_root,
            try_interactive=false,
        )
    end
end

nothing
