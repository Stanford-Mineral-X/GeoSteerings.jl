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
using Plots
using Test
using Revise

export 
    Cell, 
    Action,
    State,
    Observation,
    GeoSteeringMDP,
    generate_all_zones,
    get_amplitude_and_frequency,
    is_in_target_zone,
    get_target_bounds,
    is_required_for_connection,
    is_in_target_or_connection_needed,
    inbounds,
    initialize_mdp,
    move,
    dir,
    aarrow,
    aind,
    GeoSteeringPOMDP,
    initialize_pomdp,
    get_surrounding_status_combinations,
    get_surrounding_status,
    very_similar
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
    create_gif_from_images,
    pomdp2mdp,
    render_full,
    render_current_pos,
    render_observation,
    create_side_by_side_gif_from_images
include("utils.jl")

include("mdp.jl")
include("pomdp.jl")

end # module GeoSteerings
