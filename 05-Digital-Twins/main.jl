
# First, generate and plot the synthetic ground truth.
using Pkg

ENV["jutuljudifilter_force_install"] = "true"

params_file = abspath(joinpath(@__DIR__, "params", "small-params.jl"))
push!(ARGS, params_file)

include("scripts/plot_ground_truth.jl")

using DrWatson

save_dir_root = plotsdir()
save_dir = relpath(save_dir_root, projectdir())
if abspath(save_dir_root) != abspath(save_dir)
    mv(save_dir_root, save_dir; force=true)
end

using Markdown

# ## Saturation
d = joinpath(save_dir, "ground_truth", filestem_gt, "states")
fig_path = Markdown.parse("""
![ground truth saturation 1]($(joinpath(d, "saturation", "01.png")))
![ground truth saturation 2]($(joinpath(d, "saturation", "02.png")))
![ground truth saturation 3]($(joinpath(d, "saturation", "03.png")))
![ground truth saturation 4]($(joinpath(d, "saturation", "04.png")))
![ground truth saturation 5]($(joinpath(d, "saturation", "05.png")))
![ground truth saturation 6]($(joinpath(d, "saturation", "06.png")))
""")

# ## Pressure change
fig_path = Markdown.parse("""
![ground truth pressure_diff 1]($(joinpath(d, "pressure_diff", "01.png")))
![ground truth pressure_diff 2]($(joinpath(d, "pressure_diff", "02.png")))
![ground truth pressure_diff 3]($(joinpath(d, "pressure_diff", "03.png")))
![ground truth pressure_diff 4]($(joinpath(d, "pressure_diff", "04.png")))
![ground truth pressure_diff 5]($(joinpath(d, "pressure_diff", "05.png")))
![ground truth pressure_diff 6]($(joinpath(d, "pressure_diff", "06.png")))
""")

# ## Saturation on top of permeability
fig_path = Markdown.parse("""
![ground truth saturation 1]($(joinpath(d, "saturation_permeability", "01.png")))
![ground truth saturation 2]($(joinpath(d, "saturation_permeability", "02.png")))
![ground truth saturation 3]($(joinpath(d, "saturation_permeability", "03.png")))
![ground truth saturation 4]($(joinpath(d, "saturation_permeability", "04.png")))
![ground truth saturation 5]($(joinpath(d, "saturation_permeability", "05.png")))
![ground truth saturation 6]($(joinpath(d, "saturation_permeability", "06.png")))
""")
