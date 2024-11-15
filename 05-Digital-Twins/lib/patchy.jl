
# This is compatible with Zygote by not mutating array.
export find_water_bottom_immutable
function find_water_bottom_immutable(m::AbstractArray{avDT,N}; eps=1e-4) where {avDT,N}
    #return the indices of the water bottom of a seismic image
    wbfunc(x) = abs(x) > eps
    m_offset = m .- m[:, 1]
    idx = findfirst.(x -> wbfunc(x), eachrow(m_offset))
    return idx
end

export FloatOrMat
FloatOrMat{T} = Union{T,AbstractMatrix{<:T}}

export compute_patchy_constants
function compute_patchy_constants(
    vₚ_sat1::FloatOrMat{T},
    ρ_sat1::FloatOrMat{T},
    ϕ::FloatOrMat{T},
    d::Tuple{T,T};
    B_fl1,
    B_fl2,
    ρ_fl2,
    ρ_fl1,
) where {T}
    # vₚ_sat1 is the P-wave (m/s) speed of the rock saturated with water.
    # ρ_sat1 is the density (kg/m³) of the rock saturated with water.
    # ϕ is the porosity of the rock.

    # The P-wave modulus is M = ρ vₚ², and the shear modulus is μ = ρ vₛ²,
    # where ρ is the density, vₚ is the P-wave speed, and vₛ is the S-wave
    # speed. They are related by M = B + 4/3 μ, where B is the bulk modulus.
    # Here, we assume the S-wave speed is given by vₛ = vₚ/√3.

    # This works for Compass 2D model.
    cap_speed_threshold = 3500.0f0 # m/s
    cap_depth = 50.0f0 # meters
    cap_bulk_modulus = 5.0f10 # Pascals

    n = size(vₚ_sat1)

    vₛ = vₚ_sat1 ./ √(3.0f0)

    # Using the given ρ_sat1 and vₚ_sat1, we first compute the implied P-wave and shear
    # moduli, as well as the bulk moduli of the rock saturated with water.
    M_sat1 = ρ_sat1 .* (vₚ_sat1 .^ 2)
    μ_sat1 = ρ_sat1 .* (vₛ .^ 2)
    B_sat1 = M_sat1 .- 4.0f0 / 3.0f0 .* μ_sat1

    # Compute the bulk modulus of the dry rock.
    slow_mask = vₚ_sat1 .< cap_speed_threshold
    B_min = ifelse.(slow_mask, 1.2f0 .* B_sat1, cap_bulk_modulus)

    # Then we solve Gassman's equation for the bulk modulus of the rock
    # saturated with fluid 2.
    ptemp = (
        B_sat1 ./ (B_min .- B_sat1) .- B_fl1 ./ ϕ ./ (B_min .- B_fl1) .+
        B_fl2 ./ ϕ ./ (B_min .- B_fl2)
    )

    # Copy the values from a depth below the uncomformity layer to the depth
    # above the uncomformity layer. TBD: why?
    capgrid = Int(round(cap_depth / d[2]))
    idx = find_water_bottom_immutable((vₚ_sat1 .- cap_speed_threshold) .* (.!slow_mask))
    rows = 1:n[end]
    masks = [((rows .>= (idx[i] - capgrid)) .&& (rows .<= (idx[i] - 1))) for i in 1:n[1]]
    mask = hcat(masks...)
    ptemp_shifted = circshift(ptemp, (0, -capgrid))
    ptemp = ifelse.(mask', ptemp_shifted, ptemp)

    B_sat2 = B_min ./ (1.0f0 ./ ptemp .+ 1.0f0)

    # The bulk modulus of rock saturated with the second fluid should not be
    # higher than with the first fluid
    bigger_mask = B_sat2 .> B_sat1
    B_sat2 = ifelse.(bigger_mask, B_sat1, B_sat2)

    # The bulk modulus of rock saturated with the second fluid should not be smaller than very small.
    smaller_mask = B_sat2 .< 0
    B_sat2 = ifelse.(smaller_mask, B_sat1, B_sat2)

    # Compute P-wave modulus of rock saturated with second fluid.
    M_sat2 = B_sat2 + 4.0f0 / 3.0f0 * μ_sat1

    # Find water layer.
    idx_wb = maximum(find_water_bottom_immutable(vₚ_sat1 .- minimum(vₚ_sat1)))

    return M_sat1, M_sat2, idx_wb
end

export Patchy
function Patchy(
    sat2::FloatOrMat{T},
    vₚ_sat1::FloatOrMat{T},
    ρ_sat1::FloatOrMat{T},
    ϕ::FloatOrMat{T},
    d::Tuple{T,T};
    B_fl1=2.735f9,
    B_fl2=0.125f9,
    ρ_fl2=7.766f2,
    ρ_fl1=1.053f3,
    M_sat1=nothing,
    M_sat2=nothing,
    idx_wb=nothing,
) where {T}
    # sat2 is the saturation of CO2.
    n = size(vₚ_sat1)

    if isnothing(M_sat1) || isnothing(M_sat2) || isnothing(idx_wb)
        M_sat1, M_sat2, idx_wb = compute_patchy_constants(
            vₚ_sat1, ρ_sat1, ϕ, d; B_fl1, B_fl2, ρ_fl2, ρ_fl1
        )
    end

    # The P-wave modulus for intermediate saturations is computed with a
    # weighted harmonic mean.
    M_new = @. M_sat1 * M_sat2 / ((1.0f0 - sat2) * M_sat2 + sat2 * M_sat1)

    # Do not let the modulus change in the ocean.
    idx_wb_mask = (1:n[end] .<= idx_wb)
    idx_wb_mask_full = repeat(idx_wb_mask', n[1])
    M_new = ifelse.(idx_wb_mask_full, M_sat1, M_new)

    # The density for intermediate saturations is a weighted arithmetic mean.
    ρ_new = @. ρ_sat1 + ϕ * sat2 * (ρ_fl2 - ρ_fl1)
    ρ_new = ifelse.(idx_wb_mask_full, ρ_sat1, ρ_new)

    # Compute new P-wave speed based on P-wave modulus.
    vₚ_new = ifelse.(M_new .< 0, T(NaN), sqrt.(abs.(M_new) ./ ρ_new))
    return vₚ_new, ρ_new
end

export PatchyModel
struct PatchyModel
    vel
    rho
    phi
    boundary_mask
    d
    patchy_kwargs
end

function PatchyModel(vel, rho, phi, patchy_params, mesh_params; idx_wb=-1)
    d = Tuple(Float32.(mesh_params.d))
    bulk_H2O = Float32(patchy_params.bulk_H2O)
    bulk_CO2 = Float32(patchy_params.bulk_CO2)
    ρH2O = Float32(patchy_params.density_H2O)
    ρCO2 = Float32(patchy_params.density_CO2)

    patchy_constant_kwargs = (; B_fl1=bulk_H2O, ρ_fl1=ρH2O, B_fl2=bulk_CO2, ρ_fl2=ρCO2)
    M_sat1, M_sat2, idx_wb1 = compute_patchy_constants(
        vel, rho, phi, d; patchy_constant_kwargs...
    )
    if idx_wb == -1
        idx_wb = idx_wb1
    end
    boundary_mask = (phi .> 1e1)
    patchy_kwargs = (; patchy_constant_kwargs..., M_sat1, M_sat2, idx_wb)
    return PatchyModel(vel, rho, phi, boundary_mask, d, patchy_kwargs)
end

function (P::PatchyModel)(saturation)
    v_t, rho_t = Patchy(Float32.(saturation), P.vel, P.rho, P.phi, P.d; P.patchy_kwargs...)
    return v_t, rho_t
end
