if abspath(PROGRAM_FILE) == @__FILE__
    params_file = abspath(ARGS[1])
end

include("install.jl")

using TerminalLoggers: TerminalLogger
using Logging: global_logger
using ProgressLogging: @progress
isinteractive() && global_logger(TerminalLogger())

using DrWatson: wsave, datadir, produce_or_load, projectdir, srcdir
using Ensembles:
    Ensembles, Ensemble, get_state_keys, get_ensemble_matrix, split_clean_noisy, xor_seed!
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

mutable struct FileSampler
    field
    permutation
    i
end

function make_sampler(seed, mesh_options::MeshOptions, prior::FieldOptions)
    rng = Random.MersenneTwister(seed)
    field = create_field(mesh_options.n, prior.suboptions)
    if ndims(field) == 2
        field = reshape(field, 1, size(field)...)
    end
    permutation = Random.randperm(rng, size(field, 1))
    return FileSampler(field, permutation, 0)
end

function generate_prior_sample(sampler::FileSampler)
    if sampler.i >= length(sampler.permutation)
        sampler.i = 0
        # error("There are only $(length(sampler.permutation)) samples available")
    end
    sampler.i += 1
    return sampler.field[sampler.permutation[sampler.i], :, :]
end

struct GaussianSampler
    mean
    std
    shape
    rng
end

function make_sampler(seed, mesh_options::MeshOptions, prior::GaussianPriorOptions)
    return GaussianSampler(
        prior.mean, prior.std, mesh_options.n, Random.MersenneTwister(seed)
    )
end

function generate_prior_sample(sampler::GaussianSampler)
    return sampler.mean .+ sampler.std * randn(sampler.rng, sampler.shape)
end

function generate_initial_ensemble(params_en)
    seed = params_en.seed
    ensemble_size = params_en.size
    prior = params_en.prior

    members = [Dict{Symbol,Any}() for _ in 1:ensemble_size]

    for (k, opt) in pairs(prior)
        sampler = make_sampler(seed ⊻ hash(k), params_en.mesh, opt)
        for member in members
            member[k] = generate_prior_sample(sampler)
        end
    end

    for member in members
        member[:Permeability] = Kto3(
            member[:Permeability]; kvoverkh=params_en.permeability_v_over_h
        )

        # K_min = 500.0
        # K_minned = max.(K_min, K./mD_to_meters2)
        # K_blur = imfilter(K_minned, Kernel.gaussian(10));
        # K_lower_mD = max.(K_blur .- K_min, 0)
        # idx = find_water_bottom_immutable(K_lower_mD)
        # rows = 1:n[end]
        # masks = [rows .>= idx[i] for i = 1:n[1]];
        # update_mask = hcat(masks...)';
    end

    ensemble = Ensemble(members)
    return Dict("ensemble" => ensemble)
end

function initial_ensemble_stem(params)
    return string(hash(params.ensemble); base=62)
end

function produce_or_load_initial_ensemble(params; kwargs...)
    params_en = params.ensemble
    filestem = initial_ensemble_stem(params)

    params_file = datadir("initial_ensemble", "params", "$filestem.jld2")
    wsave(params_file; params=params_en)

    params_file = datadir("initial_ensemble", "params", "$filestem-human.yaml")
    YAML.write_file(params_file, to_dict(params_en, YAMLStyle))

    savedir = datadir("initial_ensemble", "data")
    data, filepath = produce_or_load(
        generate_initial_ensemble,
        params_en,
        savedir;
        filename=filestem,
        verbose=false,
        tag=false,
        loadfile=false,
        kwargs...,
    )
    return data, filepath, filestem
end

if abspath(PROGRAM_FILE) == @__FILE__
    params_file = abspath(ARGS[1])
    params = include(params_file)
    produce_or_load_initial_ensemble(params)
end
