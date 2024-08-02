using Test
using Random
using POMDPs
using POMDPTools
using POMDPModels
using Parameters
using StaticArrays
using Distributions
using Colors

include("../src/geosteering.jl")

# Define a helper function to create a test GeoSteeringMDP
function create_test_mdp()
	return initialize_mdp(size = (3, 3), base_amplitude = 1.0, base_frequency = 1.0,
		amplitude_variation = 0.01, frequency_variation = 0.01, phase = 0.2,
		vertical_shift = 1.0, target_thickness = 1.0, drift_prob = 0.2,
		discount = 0.9, reward_target = 50.0, reward_offtarget = -50.0, reward_goal = 1000.0,
		rng = Random.GLOBAL_RNG)
end

mdp = create_test_mdp()

# Test generate_all_zones
@testset "Test generate_all_zones" begin
	# mdp = create_test_mdp()
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
	(amplitude, frequency) = get_amplitude_and_frequency(mdp, 5)

	@test isapprox(amplitude, mdp.base_amplitude + mdp.amplitude_variation * sin(2 * pi * 5 / mdp.size[1]))
	@test isapprox(frequency, mdp.base_frequency + mdp.frequency_variation * sin(2 * pi * 5 / mdp.size[1]))
end

# Test is_in_target_zone
@testset "Test is_in_target_zone" begin
	target_zone_check = is_in_target_zone(mdp, 1, 2)
	target_off_zone_check = is_in_target_zone(mdp, 1, 3)

	@test target_zone_check == true
	@test target_off_zone_check == false
end


# Test get_target_bounds
@testset "Test get_target_bounds" begin
	(y_min, y_max) = get_target_bounds(mdp, 3)
	@test y_min <= y_max
	@test y_min in 1:mdp.size[2]
	@test y_max in 1:mdp.size[2]
end

# Test is_required_for_connection
@testset "Test is_required_for_connection" begin
	@test is_required_for_connection(mdp, 2, 2) == true
	@test is_required_for_connection(mdp, 1, 3) == false
end


# Test is_in_target_or_connection_needed
@testset "Test is_in_target_or_connection_needed" begin
	@test is_in_target_or_connection_needed(mdp, 1, 1) == false
	@test is_in_target_or_connection_needed(mdp, 2, 2) == true
end


# Test inbounds
@testset "Test inbounds" begin
	@test inbounds(mdp, Cell(5, 6)) == false
	@test inbounds(mdp, Cell(1, 2)) == true
end

# Test State + Action
@testset "Test State + Action" begin
	surrounding = get_surrounding_status(mdp, Cell(1, 1))
	state = State(cell = Cell(1, 1), is_surrounding_target = surrounding)
	next_state = move(mdp, state, RIGHT)

	@test next_state.cell == Cell(2, 1)
	@test next_state.is_surrounding_target[Cell(1, 2)] == true
end

# Test initialize_mdp
@testset "Test initialize_mdp" begin

	@test mdp.size == (3, 3)

end
