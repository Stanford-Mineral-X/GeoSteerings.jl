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
using IterTools

const Cell=SVector{2,Int}

@enum Action UP = 1 DOWN = 2 RIGHT = 3 #LEFT = 4

@enum Direction DIR_TOP = 1 DIR_BOTTOM = 2 DIR_RIGHT = 3 DIR_LEFT = 4 DIR_TOPRIGHT = 5 DIR_TOPLEFT = 6 DIR_BOTTOMRIGHT = 7 DIR_BOTTOMLEFT = 8

@with_kw mutable struct State
    cell::Cell
    is_surface_visited::Vector{Bool}
    is_surrounding_target::Dict{Cell, Bool}
end

Base.:(==)(s1::State, s2::State) = s1.cell == s2.cell && s1.is_surface_visited == s2.is_surface_visited && s1.is_surrounding_target == s2.is_surrounding_target

@with_kw mutable struct Observation
    cell::Cell
    is_surrounding_target::Dict{Cell, Bool}
end



const dir = Dict(UP=>Cell(0,1), DOWN=>Cell(0,-1), RIGHT=>Cell(1,0))#LEFT=>State(-1,0), )
const aarrow = Dict(UP=>"↑", DOWN=>"↓",  RIGHT=>"→")#, LEFT=>"←")
const aind = Dict(UP=>1, DOWN=>2, RIGHT=>3) #LEFT=>3, 
const SURROUNDINGS = Dict(
    DIR_TOP=>Cell(0,1),
    DIR_BOTTOM=>Cell(0,-1),
    DIR_RIGHT=>Cell(1,0),
    DIR_LEFT=>Cell(-1,0),
    DIR_TOPRIGHT=>Cell(1,1),
    DIR_TOPLEFT=>Cell(-1,1),
    DIR_BOTTOMRIGHT=>Cell(1,-1),
    DIR_BOTTOMLEFT=>Cell(-1,-1)
)


@with_kw mutable struct GeoSteeringMDP <: MDP{State, Action}
    size::Tuple{Int, Int}           # num_x, num_y
    base_amplitude::Float64         # Base amplitude of the sinusoidal function
    base_frequency::Float64         # Base frequency of the sinusoidal function
    amplitude_variation::Float64    # Variation in amplitude
    frequency_variation::Float64    # Variation in frequency
    target_thickness::Float64      # Thickness of the target zone
    phase::Float64                  # Phase shift of the sinusoidal function
    vertical_shift::Float64         # Vertical shift of the sinusoidal function
    drift_prob::Float64             # Probability of drifting
    discount::Float64               # Discount factor
    reward_target::Float64         # Reward for staying within the target zone
    reward_offtarget::Float64      # Reward for getting out of the target zone
    rng::AbstractRNG                # Random number generator
    target_zone::Set{Cell}         = Set{Cell}() # Set of cells in the target zone
    shale_zone::Set{Cell}           = Set{Cell}() # Set of cells in the shale zone
    nontarget_zone::Set{Cell}      = Set{Cell}() # Set of cells in the non-target zone
    terminal_zone::Set{Cell}        = Set{Cell}() # Set of cells in the terminal zone
end


@with_kw mutable struct GeoSteeringPOMDP <: POMDP{State, Action, Observation}
    size::Tuple{Int, Int}           # num_x, num_y
    base_amplitude::Float64         # Base amplitude of the sinusoidal function
    base_frequency::Float64         # Base frequency of the sinusoidal function
    amplitude_variation::Float64    # Variation in amplitude
    frequency_variation::Float64    # Variation in frequency
    phase::Float64                  # Phase shift of the sinusoidal function
    vertical_shift::Float64         # Vertical shift of the sinusoidal function
    target_thickness::Float64      # Thickness of the target zone
    drift_prob::Float64             # Probability of drifting
    discount::Float64               # Discount factor
    reward_target::Float64         # Reward for staying within the target zone
    reward_offtarget::Float64      # Reward for getting out of the target zone
    rng::AbstractRNG                # Random number generator
    target_zone::Set{Cell}         = Set{Cell}() # Set of cells in the target zone
    shale_zone::Set{Cell}           = Set{Cell}() # Set of cells in the shale zone
    nontarget_zone::Set{Cell}      = Set{Cell}() # Set of cells in the non-target zone
    terminal_zone::Set{Cell}        = Set{Cell}() # Set of cells in the terminal zone
end

function generate_all_zones(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP})
    # Generate target, non-target, and shale zones
    target_zone = Set{Cell}()
    nontarget_zone = Set{Cell}()
    shale_zone = Set{Cell}()
    terminal_zone = Set{Cell}()
    
    for x in 1:mdp.size[1], y in 1:mdp.size[2]
        if is_in_target_or_connection_needed(mdp, x, y)
            push!(target_zone, Cell(x,y))
        else
            push!(nontarget_zone, Cell(x,y))
        end
    end
    
    # Calculate shale zone based on target zone
    for s in target_zone
        x, y = s
        # Add two Cells above and below the target zone to shale zone
        for dy in -3:3
            if y + dy > 0 && y + dy <= mdp.size[2] && !(dy == 0)
                push!(shale_zone, Cell(x, y + dy))
            end
        end
    end

    # Find terminal Cells
    for y in 1:mdp.size[2]
        if Cell(mdp.size[1], y) in target_zone
            push!(terminal_zone, Cell(mdp.size[1], y))
        end
    end
    
    return (target=target_zone, nontarget=nontarget_zone, shale=shale_zone, terminal_zone = terminal_zone)
end



function get_amplitude_and_frequency(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, x::Int)
    x_factor = sin(2 * pi * x / mdp.size[1])
    amplitude = mdp.base_amplitude + mdp.amplitude_variation * x_factor
    frequency = mdp.base_frequency + mdp.frequency_variation * x_factor
    return (amplitude, frequency)
end


function is_in_target_zone(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, x::Int, y::Int)
    amplitude, frequency = get_amplitude_and_frequency(mdp, x)
    sinusoidal_y = amplitude * sin(frequency * x + mdp.phase) + mdp.vertical_shift
    half_thickness = mdp.target_thickness / 2
    lower_bound = sinusoidal_y - half_thickness
    upper_bound = sinusoidal_y + half_thickness
    return lower_bound <= y <= upper_bound
end


function get_target_bounds(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, x::Int)
    # Find indices of y-values in the target zone
    y_targets = findall(y -> is_in_target_zone(mdp, x, y), 1:mdp.size[2])
    
    # Determine the minimum and maximum y-values in the target zone
    y_min_target = minimum(y_targets)
    y_max_target = maximum(y_targets)
    
    return (y_min_target, y_max_target)
end

function is_required_for_connection(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, x::Int, y::Int)
    y_min_prev, y_max_prev = get_target_bounds(mdp, x - 1)
    y_min, y_max = get_target_bounds(mdp, x)
    if y_max_prev <= y <= y_min || y_max <= y <= y_min_prev
        return true        
    end
    return false
end

function is_in_target_or_connection_needed(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, x::Int, y::Int)
    #if the target zones are not connected, ensure connection
    if x > 1
        return is_in_target_zone(mdp, x, y) || is_required_for_connection(mdp, x, y)
    end

    return is_in_target_zone(mdp, x, y)
end

function inbounds(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, cell::Cell)    
    # return (1 <= cell[1] <= mdp.size[1] && 1 <= cell[2] <= mdp.size[2]) || 
    return cell in mdp.shale_zone || cell in mdp.target_zone
end

function get_surrounding_status(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, cell::Cell)

    surrounding_target = Dict{Cell, Bool}()
    for (dir, dcell) in SURROUNDINGS
        surrounding_target[cell + dcell] = (cell + dcell) in mdp.target_zone
    end
    return surrounding_target
end



function move(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, s::State, a::Action)
    next_cell = s.cell + dir[a]
    out_of_bound_x = next_cell[1] < 1 || next_cell[1] > length(s.is_surface_visited)
    out_of_bound_y = next_cell[2] < 1 || next_cell[2] > length(s.is_surface_visited)
    if out_of_bound_x || out_of_bound_y 
        return s
    end
    next_is_surface_visited = collect([b_ for b_ in s.is_surface_visited])
    next_is_surface_visited[1:s.cell.x + dir[a].x] .= true
    next_surrounding_target = get_surrounding_status(mdp, next_cell)
    
    return State(cell=next_cell, is_surface_visited=next_is_surface_visited, is_surrounding_target=next_surrounding_target)   
end
    
function initialize_mdp(;
    size::Tuple{Int, Int}           = (15, 15),
    base_amplitude::Float64         = 3.0,
    base_frequency::Float64         = 1.0,  # Base frequency of the sinusoidal function
    amplitude_variation::Float64    = 0.5,  # Variation in amplitude
    frequency_variation::Float64    = 0.05,  # Variation in frequency
    phase::Float64                  = 0.3, # Phase shift of the sinusoidal function
    vertical_shift::Float64         = 7.0,  # Vertical shift of the sinusoidal function
    target_thickness::Float64      = 5.0,  # Thickness of the target zone
    drift_prob::Float64             = 0.3,  # Probability of drifting
    discount::Float64               = 0.95, # Discount factor
    reward_target::Float64         = 0.0,
    reward_offtarget::Float64      = -100.0,
    rng::AbstractRNG                = Random.GLOBAL_RNG
)
    mdp = GeoSteeringMDP(
        size=size,
        base_amplitude=base_amplitude,
        base_frequency=base_frequency,
        amplitude_variation=amplitude_variation,
        frequency_variation=frequency_variation,
        phase=phase,
        vertical_shift=vertical_shift,
        target_thickness=target_thickness,
        drift_prob=drift_prob,
        discount=discount,
        reward_target=reward_target,
        reward_offtarget=reward_offtarget,
        rng=rng
    )
    mdp.target_zone, mdp.nontarget_zone, mdp.shale_zone, mdp.terminal_zone = generate_all_zones(mdp)    
    return mdp
end


@with_kw mutable struct GeoSteeringPOMDP <: POMDP{State, Action, Observation}
    size::Tuple{Int, Int}           # num_x, num_y
    base_amplitude::Float64         # Base amplitude of the sinusoidal function
    base_frequency::Float64         # Base frequency of the sinusoidal function
    amplitude_variation::Float64    # Variation in amplitude
    frequency_variation::Float64    # Variation in frequency
    phase::Float64                  # Phase shift of the sinusoidal function
    vertical_shift::Float64         # Vertical shift of the sinusoidal function
    target_thickness::Float64      # Thickness of the target zone
    drift_prob::Float64             # Probability of drifting
    discount::Float64               # Discount factor
    reward_target::Float64         # Reward for staying within the target zone
    reward_offtarget::Float64      # Reward for getting out of the target zone
    rng::AbstractRNG                # Random number generator
    target_zone::Set{Cell}         = Set{Cell}() # Set of cells in the target zone
    shale_zone::Set{Cell}           = Set{Cell}() # Set of cells in the shale zone
    nontarget_zone::Set{Cell}      = Set{Cell}() # Set of cells in the non-target zone
    terminal_zone::Set{Cell}        = Set{Cell}() # Set of cells in the terminal zone
end


function initialize_pomdp(;
    size::Tuple{Int, Int}           = (5, 5),
    base_amplitude::Float64         = 1.0,
    base_frequency::Float64         = 1.0,  # Base frequency of the sinusoidal function
    amplitude_variation::Float64    = 0.5,  # Variation in amplitude
    frequency_variation::Float64    = 0.05,  # Variation in frequency
    phase::Float64                  = 0.3, # Phase shift of the sinusoidal function
    vertical_shift::Float64         = 3.0,  # Vertical shift of the sinusoidal function
    target_thickness::Float64      = 2.0,  # Thickness of the target zone
    drift_prob::Float64             = 0.3,  # Probability of drifting
    discount::Float64               = 0.95, # Discount factor
    reward_target::Float64         = 100.0,
    reward_offtarget::Float64      = -100.0,
    rng::AbstractRNG                = Random.GLOBAL_RNG
)
    pomdp = GeoSteeringPOMDP(
        size=size,
        base_amplitude=base_amplitude,
        base_frequency=base_frequency,
        amplitude_variation=amplitude_variation,
        frequency_variation=frequency_variation,
        phase=phase,
        vertical_shift=vertical_shift,
        target_thickness=target_thickness,
        drift_prob=drift_prob,
        discount=discount,
        reward_target=reward_target,
        reward_offtarget=reward_offtarget,
        rng=rng
    )
    pomdp.target_zone, pomdp.nontarget_zone, pomdp.shale_zone, pomdp.terminal_zone = generate_all_zones(pomdp)    
    return pomdp
end