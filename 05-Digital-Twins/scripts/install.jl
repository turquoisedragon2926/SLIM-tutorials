
macro codegen_copy_constructor(T)
    esc(
        quote
            function $T(x::$T; kwargs...)
                default_kwargs = (f => getfield(x, f) for f in fieldnames($T))
                return $T(; default_kwargs..., kwargs...)
            end
        end,
    )
end

if isinteractive()
    using Pkg: Pkg
    try
        using Revise
    catch
        using Revise
        Pkg.add("Revise")
    end
end

if get(ENV, "jutuljudifilter_force_install", "false") == "true" ||
    basename(dirname(Base.active_project())) in ["v1.11", "v1.10"]
    using Pkg: Pkg

    Pkg.activate(joinpath(@__DIR__, ".."))
    @assert basename(dirname(Base.active_project())) == "05-Digital-Twins"

    try
        using JutulJUDIFilter: JutulJUDIFilter
    catch
        path = get(ENV, "jutuljudifilter_path", joinpath(@__DIR__, "..", "..", ".."))
        Pkg.add(; url="https://github.com/DataAssimilation/JutulJUDIFilter.jl")
        using JutulJUDIFilter: JutulJUDIFilter
    end

    try
        using Ensembles: Ensembles
    catch
        JutulJUDIFilter.install(:Ensembles)
        using Ensembles: Ensembles
    end

    try
        import ConfigurationsJutulDarcy: ConfigurationsJutulDarcy
    catch
        JutulJUDIFilter.install(:ConfigurationsJutulDarcy)
    end

    # try
    #     import ConfigurationsJUDI: ConfigurationsJUDI
    # catch
    #     JutulJUDIFilter.install(:ConfigurationsJUDI)
    # end

    try
        using EnsembleKalmanFilters: EnsembleKalmanFilters
    catch
        Ensembles.install(:EnsembleKalmanFilters)
    end

    try
        using NormalizingFlowFilters: NormalizingFlowFilters
    catch
        Ensembles.install(:NormalizingFlowFilters)
    end

    Pkg.add([
        "CairoMakie",
        "ChainRulesCore",
        "Configurations",
        "Distributed",
        "DrWatson",
        "Format",
        "ImageFiltering",
        "JLD2",
        "JUDI",
        "JutulDarcy",
        "LinearAlgebra",
        "Logging",
        "Makie",
        "Markdown",
        "ProgressLogging",
        "Random",
        "Statistics",
        "TerminalLoggers",
        "WGLMakie",
        "YAML",
    ])

    Pkg.instantiate()
end

using DrWatson: projectdir
if !(projectdir("lib") in LOAD_PATH)
    push!(LOAD_PATH, projectdir("lib"))
end
