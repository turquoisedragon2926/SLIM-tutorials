using Configurations: @option

using Ensembles: Ensembles, AbstractNoisyOperator
using ImageFiltering: imfilter, Kernel
using JUDI: find_water_bottom
using DrWatson: projectdir

using FilterComparison
include("seismic_model.jl")

struct SeismicCO2Observer{I} <: AbstractNoisyOperator
    M::SeismicModel
    P::PatchyModel
end

function (S::SeismicCO2Observer{false})(member::Dict)
    dshot, rtm, dshot_noisy, rtm_noisy = S(member[:Saturation])
    obs = Dict(
        :dshot => deepcopy(dshot.data),
        :rtm => deepcopy(rtm),
        :dshot_noisy => deepcopy(dshot_noisy.data),
        :rtm_noisy => deepcopy(rtm_noisy),
    )
    return obs
end

function (S::SeismicCO2Observer{true})(member::Dict)
    (dshot, rtm, dshot_noisy, rtm_noisy), velocity, density = S(member[:Saturation])
    obs = Dict(
        :dshot => deepcopy(dshot.data),
        :rtm => deepcopy(rtm),
        :dshot_noisy => deepcopy(dshot_noisy.data),
        :rtm_noisy => deepcopy(rtm_noisy),
        :velocity => deepcopy(velocity),
        :density => deepcopy(density),
    )
    return obs
end

function Ensembles.split_clean_noisy(M::SeismicCO2Observer{false}, obs::Dict{Symbol,<:Any})
    obs_clean = typeof(obs)()
    obs_noisy = typeof(obs)()
    for key in (:dshot, :rtm)
        obs_clean[key] = obs[key]
        obs_noisy[key] = obs[Symbol(key, :_noisy)]
    end
    return obs_clean, obs_noisy
end

function Ensembles.split_clean_noisy(M::SeismicCO2Observer{true}, obs::Dict{Symbol,<:Any})
    obs_clean = typeof(obs)(:velocity => obs[:velocity], :density => obs[:density])
    obs_noisy = typeof(obs)()
    for key in (:dshot, :rtm)
        obs_clean[key] = obs[key]
        obs_noisy[key] = obs[Symbol(key, :_noisy)]
    end
    return obs_clean, obs_noisy
end

function (S::SeismicCO2Observer{false})(saturation::AbstractArray)
    saturation = reshape(saturation, size(S.M.model))
    saturation = ifelse.(S.P.boundary_mask, 0, saturation)
    v_t, rho_t = S.P(saturation)
    return S.M(v_t, rho_t)
end

function (S::SeismicCO2Observer{true})(saturation::AbstractArray)
    saturation = reshape(saturation, size(S.M.model))
    saturation = ifelse.(S.P.boundary_mask, 0, saturation)
    v_t, rho_t = S.P(saturation)
    obs = S.M(v_t, rho_t)
    return obs, v_t, rho_t
end

function SeismicCO2Observer(options::SeismicCO2ObserverOptions)
    M = SeismicModel(options.seismic)
    porosity = create_field(options.seismic.mesh, options.rock_physics.porosity)
    porosity = Float32.(porosity)
    # Patchy model uses true velocity and density.
    P = PatchyModel(M.vel, M.rho, porosity, options.rock_physics, options.seismic.mesh)
    return SeismicCO2Observer{options.save_intermediate}(M, P)
end

function create_velocity_field(mesh::MeshOptions, options::FieldOptions)
    return create_field(mesh, options)
end
function create_velocity_field(mesh::MeshOptions, options)
    return create_velocity_field(mesh, options, Val(options.type))
end
function create_velocity_field(mesh::MeshOptions, options, ::Val{:squared_slowness})
    return (1 ./ create_field(mesh, options.field)) .^ 0.5
end

function create_background_velocity_field(
    velocity, mesh::MeshOptions, options::FieldOptions
)
    return create_field(mesh, options)
end
function create_background_velocity_field(
    velocity, mesh::MeshOptions, options::BackgroundBlurOptions
)
    v0 = deepcopy(velocity)
    idx_wb = find_water_bottom(v0)
    rows = 1:mesh.n[end]
    masks = [rows .> idx_wb[i] for i in 1:mesh.n[1]]
    mask = hcat(masks...)'
    v0[mask] = (1.0f0 ./ imfilter(1.0f0 ./ v0, Kernel.gaussian(options.cells)))[mask]
    return v0
end

function create_background_density_field(density, mesh::MeshOptions, options::FieldOptions)
    return create_field(mesh, options)
end
function create_background_density_field(
    density, mesh::MeshOptions, options::BackgroundBlurOptions
)
    rho0 = deepcopy(density)
    idx_wb = find_water_bottom(rho0)
    rows = 1:mesh.n[end]
    masks = [rows .> idx_wb[i] for i in 1:mesh.n[1]]
    mask = hcat(masks...)'
    rho0[mask] = (1.0f0 ./ imfilter(1.0f0 ./ rho0, Kernel.gaussian(options.cells)))[mask]
    return rho0
end

function read_static_seismic_params(options::SeismicObserverOptions)
    velocity = create_velocity_field(options.mesh, options.velocity)
    density = create_field(options.mesh, options.density)
    velocity0 = create_background_velocity_field(
        velocity, options.mesh, options.background_velocity
    )
    density0 = create_background_density_field(
        density, options.mesh, options.background_density
    )
    return (; velocity, density, velocity0, density0)
end

function SeismicModel(options::SeismicObserverOptions)
    (; velocity, density, velocity0, density0) = read_static_seismic_params(options)
    return SeismicModel(
        velocity,
        density,
        velocity0,
        density0;
        options.mesh.n,
        options.mesh.d,
        dtR=options.dtR,
        timeR=options.timeR,
        f0=options.f0,
        nbl=options.nbl,
        snr=options.snr,
        seed=options.seed,
        depth_scaling_exponent=options.depth_scaling_exponent,
        observation_type=options.type,
        source_receiver_geometry=options.source_receiver_geometry,
    )
end

function FilterComparison.get_observer(options::SeismicCO2ObserverOptions)
    return SeismicCO2Observer(options)
end

function Ensembles.xor_seed!(M::SeismicCO2Observer, seed_mod::UInt)
    return Ensembles.xor_seed!(M.M, seed_mod)
end

function Ensembles.xor_seed!(M::SeismicModel, seed_mod::UInt)
    return Random.seed!(M.rng, xor(M.seed, seed_mod))
end
