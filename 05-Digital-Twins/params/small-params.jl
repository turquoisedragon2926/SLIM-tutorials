# Define parameters.

include("../scripts/install.jl")

using ConfigurationsJutulDarcy
using ConfigurationsJutulDarcy: @option
using ConfigurationsJutulDarcy: SVector
using JutulDarcy.Jutul

using DrWatson
using FilterComparison

Darcy, bar, kg, meter, day, yr = si_units(:darcy, :bar, :kilogram, :meter, :day, :year)
mD_to_meters2 = 1e-3 * Darcy

injection_well_trajectory = (
    SVector(1875.0, 50.0, 1775.0),  # First point
    SVector(1875.0, 50.0, 1775.0 + 37.5),  # Second point
)

simple_system = CO2BrineSimpleOptions(;
    viscosity_CO2=1e-4,
    viscosity_H2O=1e-3,
    density_CO2=501.9,
    density_H2O=1053.0,
    reference_pressure=1.5e7,
    compressibility_CO2=8e-9,
    compressibility_H2O=3.6563071e-10,
)

params_transition = JutulOptions(;
    mesh=MeshOptions(; n=(25, 1, 25), d=(162.5, 1e2, 86.25)),
    system=simple_system,
    porosity=FieldOptions(0.25),
    permeability=FieldOptions(0.1Darcy),
    permeability_v_over_h=0.36,
    temperature=FieldOptions(convert_to_si(30.0, :Celsius)),
    rock_density=FieldOptions(30.0),
    rock_heat_capacity=FieldOptions(900.0),
    rock_thermal_conductivity=FieldOptions(3),
    fluid_thermal_conductivity=FieldOptions(0.6),
    component_heat_capacity=FieldOptions(4184.0),
    injection=WellOptions(; trajectory=injection_well_trajectory, name=:Injector),
    time=(
        TimeDependentOptions(;
            years=10.0,
            controls=(
                WellRateOptions(;
                    type="injector",
                    name=:Injector,
                    fluid_density=501.9,
                    rate_mtons_year=0.8,
                ),
            ),
        ),
        TimeDependentOptions(; years=475.0, controls=()),
    ),
)

observer_options = NoisyObservationOptions(; noise_scale=1.0, keys=(:Saturation,))

ground_truth = ModelOptions(;
    transition=params_transition,
    observation=MultiTimeObserverOptions(;
        observers=(
            3 * 0yr => observer_options,
            3 * 1yr => observer_options,
            3 * 2yr => observer_options,
            3 * 3yr => observer_options,
            3 * 4yr => observer_options,
            3 * 5yr => observer_options,
        ),
    ),
)

params = JutulJUDIFilterOptions(;
    ground_truth,
    ensemble=EnsembleOptions(;
        size=10,
        seed=9347215,
        mesh=params_transition.mesh,
        permeability_v_over_h=0.36,
        prior=(;
            Saturation=GaussianPriorOptions(; mean=0, std=0),
            Permeability=FieldOptions(;
                suboptions=FieldFileOptions(;
                    file="/opt/SLIM-Storage/compass_small/perm_poro.jld2",
                    key="Ks",
                    scale=mD_to_meters2,
                    resize=true,
                ),
            ),
        ),
    ),
    estimator=EstimatorOptions(;
        transition=ground_truth.transition,
        observation=ground_truth.observation,
        # algorithm=nothing,
        assimilation_state_keys=(:Saturation,),
        assimilation_obs_keys=(:rtm,),
        algorithm=EnKFOptions(;
            noise=NoiseOptions(; std=1, type=:diagonal),
            include_noise_in_obs_covariance=false,
            rho=0,
        ),
    ),
)
