
if get(ENV, "jutuljudifilter_force_install", "false") == "true" ||
    basename(dirname(Base.active_project())) != "05-Digital-Twins"
    using Pkg: Pkg

    if basename(dirname(Base.active_project())) != "05-Digital-Twins"
        Pkg.activate(joinpath(@__DIR__, ".."))
    end
    @assert basename(dirname(Base.active_project())) == "05-Digital-Twins"

    try
        using JutulJUDIFilter: JutulJUDIFilter
    catch
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

    try
        using EnsembleKalmanFilters: EnsembleKalmanFilters
    catch
        Ensembles.install(:EnsembleKalmanFilters)
    end

    Pkg.add([
        "CairoMakie",
        "ChainRulesCore",
        "Configurations",
        "DrWatson",
        "Format",
        "ImageFiltering",
        "ImageTransformations",
        "JLD2",
        "JUDI",
        "JutulDarcy",
        "LinearAlgebra",
        "Logging",
        "Makie",
        "ProgressLogging",
        "Random",
        "Statistics",
        "TerminalLoggers",
        "YAML",
    ])

    Pkg.instantiate()
end

using DrWatson: projectdir
if !(projectdir("lib") in LOAD_PATH)
    push!(LOAD_PATH, projectdir("lib"))
end
