using JUDI
using LinearAlgebra: norm
using Random
using ChainRulesCore: ChainRulesCore

FT = Float32

include("seismic_utils.jl")

Base.@kwdef struct UnitConverter
    time::FT
    area::FT
    volume::FT
    acceleration::FT
    force::FT
    distance::FT
    mass::FT
    velocity::FT
    specific_acoustic_impedance::FT
    density::FT
    pressure::FT
end

function InitializeUnitConverter(; time, distance, mass)
    time = FT(time)
    distance = FT(distance)
    mass = FT(mass)

    area = distance * distance
    volume = distance * area
    density = mass / volume

    velocity = distance / time
    acceleration = velocity / time
    force = mass * acceleration
    pressure = force / area

    specific_acoustic_impedance = velocity * density

    kwargs = (;
        time,
        distance,
        mass,
        area,
        volume,
        density,
        velocity,
        acceleration,
        force,
        pressure,
        specific_acoustic_impedance,
    )
    return UnitConverter(; kwargs...)
end

# JUDI uses the following base units: meters, milliseconds, and megagrams.
const SI_to_JUDI = InitializeUnitConverter(; distance=1, time=1e3, mass=1e-3)
const JUDI_to_SI = InitializeUnitConverter(; distance=1, time=1e-3, mass=1e3)

function JUDI._worker_pool()
    return nothing
end

struct SeismicModel{A,B,C,D,E,F,G}
    model::A
    q::B
    Tm::C
    S::G
    vel::Union{FT,Matrix{FT}}
    rho::Union{FT,Matrix{FT}}
    imp::Union{FT,Matrix{FT}}
    F::D
    J::E
    vel0::Union{FT,Matrix{FT}}
    rho0::Union{FT,Matrix{FT}}
    imp0::Union{FT,Matrix{FT}}
    F0::D
    J0::E
    observation_type::Val{F}
    snr::FT
    seed::UInt64
    rng
end

function SeismicModel(vel, rho, vel0, rho0; kwargs...)
    return SeismicModel(vel, rho, vel, rho)
end

# Background model is used for linearization (both forward and adjoint Jacobians).
function SeismicModel(
    vel,
    rho,
    vel0,
    rho0;
    n,
    d,
    dtR,
    timeR,
    f0,
    nbl,
    snr,
    seed,
    depth_scaling_exponent,
    observation_type=:shot,
    source_receiver_geometry,
)
    d = FT.(d)
    dtR = FT(dtR)
    timeR = FT(timeR)
    f0 = FT(f0)

    vel = FT.(vel)
    rho = FT.(rho)

    vel0 = FT.(vel0)
    rho0 = FT.(rho0)

    idx_wb = maximum(find_water_bottom(log.(vel) .- log(vel[1, 1])))
    srcGeometry, recGeometry = build_source_receiver_geometry(
        n, d, dtR, timeR, idx_wb; params=source_receiver_geometry
    )

    # Set up source term.
    wavelet = ricker_wavelet(timeR, dtR, f0)
    q = judiVector(srcGeometry, wavelet)

    # Create model.
    origin = (0, 0)
    rho_judi = rho * SI_to_JUDI.density
    vel_judi = vel * SI_to_JUDI.velocity
    rho0_judi = rho0 * SI_to_JUDI.density
    vel0_judi = vel0 * SI_to_JUDI.velocity
    m = (1 ./ vel_judi) .^ 2.0f0
    model = Model(n, d, origin, m; rho=rho_judi, nb=nbl)

    m0 = (1 ./ vel0_judi) .^ 2.0f0
    model0 = Model(n, d, origin, m0; rho=rho0_judi, nb=nbl)

    # Mute the water column and do depth scaling.
    Tm = judiTopmute(n, idx_wb, 1)
    if depth_scaling_exponent == 0
        S = 1
    else
        S = judiDepthScaling(model; K=depth_scaling_exponent)
    end

    # Set up modeling operators.
    options = Options(; IC="isic")
    F = judiModeling(model, srcGeometry, recGeometry; options)
    J = judiJacobian(F, q)

    F0 = judiModeling(model0, srcGeometry, recGeometry; options)
    J0 = judiJacobian(F0, q)

    imp = vel .* rho
    imp0 = vel0 .* rho0
    observation_type = Val(observation_type)
    rng = Random.MersenneTwister(seed)
    return SeismicModel(
        model,
        q,
        Tm,
        S,
        vel,
        rho,
        imp,
        F,
        J,
        vel0,
        rho0,
        imp0,
        F0,
        J0,
        observation_type,
        FT(snr),
        seed,
        rng,
    )
end

function SeismicModel(M::SeismicModel, vel, rho, vel0, rho0)
    srcGeometry = M.F.qInjection.op.geometry
    recGeometry = M.F.rInterpolation.geometry
    n = size(M.model)
    d = spacing(M.model)
    origin = M.model.G.o
    nbl = M.model.G.nb

    vel_judi = vel * SI_to_JUDI.velocity
    rho_judi = rho * SI_to_JUDI.density

    vel0_judi = vel0 * SI_to_JUDI.velocity
    rho0_judi = rho0 * SI_to_JUDI.density

    m = (1 ./ vel_judi) .^ 2.0f0
    m0 = (1 ./ vel0_judi) .^ 2.0f0

    model = Model(n, d, origin, m; rho=rho_judi, nb=nbl)
    model0 = Model(n, d, origin, m0; rho=rho0_judi, nb=nbl)
    options = Options(; IC="isic")
    F = judiModeling(model, srcGeometry, recGeometry; options)
    J = judiJacobian(F, M.q)
    F0 = judiModeling(model0, srcGeometry, recGeometry; options)
    J0 = judiJacobian(F0, M.q)
    imp = vel .* rho
    imp0 = vel0 .* rho0
    return SeismicModel(
        model,
        M.q,
        M.Tm,
        M.S,
        vel,
        rho,
        imp,
        F,
        J,
        vel0,
        rho0,
        imp0,
        F0,
        J0,
        M.observation_type,
        M.snr,
        M.seed,
        M.rng,
    )
end

function (M::SeismicModel)(vel, rho; kwargs...)
    return M(vel, rho, M.observation_type; kwargs...)
end

function (M::SeismicModel)(vel, rho, ::Val{:shot})
    vel_judi = vel * SI_to_JUDI.velocity
    m = (1 ./ vel_judi) .^ 2
    obs = M.F(m, M.q)
    return obs * JUDI_to_SI.pressure
end

function (M::SeismicModel)(vel, rho, ::Val{:born})
    imp = vel .* rho
    dimp = vec(imp .- M.imp0)
    # J has JUDI units of pressure per impedance.
    conversion = JUDI_to_SI.pressure / JUDI_to_SI.specific_acoustic_impedance
    return M.J0 * dimp .* conversion
end

function ChainRulesCore.rrule(
    ::typeof(*), F::JUDI.judiPropagator, x::AbstractArray{T}
) where {T}
    """The lazy evaluation in JUDI's rrule doesn't work right, so I got rid of it."""
    ra = F.options.return_array
    y = F * x
    postx = ra ? (dx -> reshape(dx, size(x))) : identity
    function Fback(Δy)
        dx = postx(F' * Δy)
        # F is m parametric
        dF = JUDI.∇prop(F, x, Δy)
        return ChainRulesCore.NoTangent(), dF, dx
    end
    y = F.options.return_array ? reshape_array(y, F) : y
    return y, Fback
end

function (M::SeismicModel)(dshot, ::Val{:rtm})
    n = size(M.model)
    rtm = M.J0' * dshot
    rtm = M.S * M.Tm * vec(rtm.data)
    rtm = reshape(rtm, n)

    # J has JUDI units of pressure per impedance.
    conversion = JUDI_to_SI.pressure / JUDI_to_SI.specific_acoustic_impedance

    # Tm is unitless and S has JUDI units of length.
    conversion *= JUDI_to_SI.distance

    return rtm * conversion
end

function (M::SeismicModel)(vel, rho, ::Val{:born_rtm_depth})
    dshot, rtm, dshot_noisy, rtm_noisy = M(vel, rho, Val(:born_shot_rtm_depth_noise))
    return rtm
end

function (M::SeismicModel)(vel, rho, ::Val{:born_rtm_depth_noise})
    dshot, rtm, dshot_noisy, rtm_noisy = M(vel, rho, Val(:born_shot_rtm_depth_noise))
    return rtm, rtm_noisy
end

function (M::SeismicModel)(vel, rho, ::Val{:born_shot_rtm_depth_noise})
    dshot = M(vel, rho, Val(:born))
    rtm = M(dshot, Val(:rtm))

    dshot = deepcopy(dshot)
    rtm = deepcopy(rtm)

    dshot_noisy = dshot + generate_noise(M, dshot, M.snr) .* norm(dshot)
    println("Noise norm: $(norm(dshot_noisy - dshot))")
    println("SNR: $(M.snr)")
    rtm_noisy = M(dshot_noisy, Val(:rtm))

    dshot_noisy = deepcopy(dshot_noisy)
    rtm_noisy = deepcopy(rtm_noisy)

    return dshot, rtm, dshot_noisy, rtm_noisy
end

function build_source_receiver_geometry(n, d, dtR, timeR, idx_wb; params)
    src_positions, rec_positions = build_source_receiver_geometry(n, d, idx_wb; params)
    (; nsrc, nrec) = params

    # srcGeometry = Geometry(convertToCell.([xsrc, ysrc, zsrc])...; dt=dtR, t=timeR)
    # recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)
    srcGeometry = Geometry(convertToCell.(src_positions)...; dt=dtR, t=timeR)
    recGeometry = Geometry(rec_positions...; dt=dtR, t=timeR, nsrc=nsrc)
    return srcGeometry, recGeometry
end

function generate_noise(M::SeismicModel, ref, snr)
    source = M.q.data[1]
    noise = deepcopy(ref)
    for noise_i in noise.data
        v = randn(FT, size(noise_i))
        noise_i .= real.(ifft(fft(v) .* fft(source)))
    end
    noise = noise / norm(noise) * 10.0f0^(-snr / 20.0f0)
    return noise
end

function compute_noise_info(params_seismic)
    timeR = FT(params_seismic[:timeR])
    dtR = FT(params_seismic[:dtR])
    f0 = FT(params_seismic[:f0])
    wavelet = ricker_wavelet(timeR, dtR, f0)
    return wavelet
end

function generate_noise(source, ref, snr)
    noise = deepcopy(ref)
    for noise_i in noise.data
        v = randn(FT, size(noise_i))
        noise_i .= real.(ifft(fft(v) .* fft(source)))
    end
    noise = noise / norm(noise) * 10.0f0^(-snr / 20.0f0)
    return noise
end
