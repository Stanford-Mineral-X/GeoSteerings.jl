using Test
using Random
using POMDPs
using POMDPTools
using POMDPModels
using Parameters
using StaticArrays
using Distributions
using Colors

# Include the code you want to test here (or use include to bring it in)
include("../src/geosteering.jl")

# Define a helper function to create a test GeoSteeringMDP
function create_test_mdp()
    return initialize_mdp(size=(10, 10), base_amplitude=2.0, base_frequency=1.0,
                          amplitude_variation=0.5, frequency_variation=0.1, phase=0.2,
                          vertical_shift=5.0, target_reward=10.0, target_thickness=3.0,
                          nontarget_penalty=-5.0, drift_prob=0.2, discount=0.9,
                          reward_target=50.0, reward_redundant_target=0.0, reward_offtarget=-50.0,
                          rng=Random.GLOBAL_RNG)
end

# Test generate_all_zones
@testset "Test generate_all_zones" begin
    mdp = create_test_mdp()
    zones = generate_all_zones(mdp)
    
    # @test typeof(zones) == NamedTuple
    @test :target in keys(zones)
    @test :nontarget in keys(zones)
    @test :shale in keys(zones)
    @test :terminal_zone in keys(zones)
    
    @test isa(zones.target, Set{Cell})
    @test isa(zones.nontarget, Set{Cell})
    @test isa(zones.shale, Set{Cell})
    @test isa(zones.terminal_zone, Set{Cell})
end

# Test get_amplitude_and_frequency
@testset "Test get_amplitude_and_frequency" begin
    mdp = create_test_mdp()
    (amplitude, frequency) = get_amplitude_and_frequency(mdp, 5)
    
    @test isapprox(amplitude, mdp.base_amplitude + mdp.amplitude_variation * sin(2 * pi * 5 / mdp.size[1]))
    @test isapprox(frequency, mdp.base_frequency + mdp.frequency_variation * sin(2 * pi * 5 / mdp.size[1]))
end

# Test is_in_target_zone
@testset "Test is_in_target_zone" begin
    mdp = create_test_mdp()
    target_zone_check = is_in_target_zone(mdp, 1, 6)
    
    @test target_zone_check == true 
end

# Test get_target_bounds
@testset "Test get_target_bounds" begin
    mdp = create_test_mdp()
    (y_min, y_max) = get_target_bounds(mdp, 5)
    
    @test y_min <= y_max
    @test y_min in 1:mdp.size[2]
    @test y_max in 1:mdp.size[2]
end

# Test is_required_for_connection
@testset "Test is_required_for_connection" begin
    mdp = create_test_mdp()
    connection_check = is_required_for_connection(mdp, 5, 6)
    
    @test connection_check == false 
end

# Test is_in_target_or_connection_needed
@testset "Test is_in_target_or_connection_needed" begin
    mdp = create_test_mdp()
    target_or_connection_check = is_in_target_or_connection_needed(mdp, 5, 6)
    
    @test target_or_connection_check == false
end

# Test inbounds
@testset "Test inbounds" begin
    mdp = create_test_mdp()
    inbounds_check = inbounds(mdp, Cell(5, 6))
    
    @test inbounds_check == true 
end

# Test State + Action
@testset "Test State + Action" begin
    state = State(cell=Cell(5, 5), is_surface_visited=Bool[false for _ in 1:10])
    next_state = state + RIGHT
    
    @test next_state.cell == Cell(6, 5)
    @test next_state.is_surface_visited[6] == true
end

# Test initialize_mdp
@testset "Test initialize_mdp" begin
    mdp = create_test_mdp()
    
    @test mdp.size == (10, 10)
    @test mdp.base_amplitude == 2.0
    @test mdp.base_frequency == 1.0
    @test mdp.amplitude_variation == 0.5
    @test mdp.frequency_variation == 0.1
    @test mdp.phase == 0.2
    @test mdp.vertical_shift == 5.0
    @test mdp.target_reward == 10.0
    @test mdp.target_thickness == 3.0
    @test mdp.nontarget_penalty == -5.0
    @test mdp.drift_prob == 0.2
    @test mdp.discount == 0.9
    @test mdp.reward_target == 50.0
    @test mdp.reward_redundant_target == 0.0
    @test mdp.reward_offtarget == -50.0
end
