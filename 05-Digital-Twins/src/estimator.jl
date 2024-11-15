using LinearAlgebra: Diagonal
using EnsembleKalmanFilters: EnKF
using Configurations: from_dict

function get_noise_covariance(params::NoiseOptions)
    type = params.type
    if type == :diagonal
        R = Float64(params.std)^2
        return R
    end
    throw(ArgumentError("Unknown noise type: $type"))
end

function get_estimator(params_estimator::EnKFOptions)
    R = get_noise_covariance(params_estimator.noise)
    estimator = EnKF(
        R, params_estimator.include_noise_in_obs_covariance, params_estimator.rho
    )
    return estimator
end

get_estimator(params_estimator::Nothing) = nothing
