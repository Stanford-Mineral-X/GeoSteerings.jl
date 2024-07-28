module GeoSteerings

using POMDPs
using POMDPTools
using POMDPModels
using Parameters
using StaticArrays
using Distributions
using Random
using Compose
using ColorSchemes
using Colors
using POMDPs
using POMDPTools
using POMDPModels
using Parameters
using Cairo
using Fontconfig
using Images
using ImageIO
using Plots
using Test
using Revise

export 
    Cell, 
    Action,
    State,
    GeoSteeringMDP,
    generate_all_zones,
    get_amplitude_and_frequency,
    is_in_target_zone,
    get_target_bounds,
    is_required_for_connection,
    is_in_target_or_connection_needed,
    inbounds,
    initialize_mdp,
    dir,
    aarrow,
    aind
include("geosteering.jl")

export 
    rgb_darkgray,
    rgb_lightgray,
    rgb_white,
    rgb_black,
    cell_ctx,
    render_cell,
    render_agent,
    render_next_agent,
    render,
    savefig,
    create_gif_from_images
include("utils.jl")

end # module
