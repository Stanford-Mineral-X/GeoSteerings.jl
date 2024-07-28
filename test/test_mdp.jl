using Test
using Random
using POMDPs
using POMDPTools
using POMDPModels
using Parameters
using StaticArrays
using Distributions
using Colors
using IterTools

# Include the code you want to test here (or use include to bring it in)
include("../src/geosteering.jl")
include("../src/mdp.jl")

# Define a helper function to create a test GeoSteeringMDP
function create_test_mdp()
    return initialize_mdp(size=(10, 10), base_amplitude=2.0, base_frequency=1.0,
                          amplitude_variation=0.5, frequency_variation=0.1, phase=0.2,
                          vertical_shift=5.0, target_reward=10.0, target_thickness=3.0,
                          nontarget_penalty=-5.0, drift_prob=0.2, discount=0.9,
                          reward_target=50.0, reward_redundant_target=0.0, reward_offtarget=-50.0,
                          rng=Random.GLOBAL_RNG)
end

# Test POMDPs.states
@testset "Test POMDPs.states" begin
    mdp = create_test_mdp()
    states = POMDPs.states(mdp)
    
    num_cells = prod(mdp.size) + 1
    num_surfaces = 2 ^ mdp.size[1]
    @test length(states) == num_cells * num_surfaces
    
    # Check some sample states
    # @test states[1].cell == Cell(1, 1)
    # @test states[end].cell == Cell(-1, -1)
end

# Test POMDPs.stateindex
@testset "Test POMDPs.stateindex" begin
    mdp = create_test_mdp()
    s = State(Cell(1, 1), [false for _ in 1:mdp.size[1]])
    index = POMDPs.stateindex(mdp, s)
    
    @test index == 1  # Adjust this based on the actual index calculation
    
    s = State(Cell(-1, -1), [true for _ in 1:mdp.size[1]])
    index = POMDPs.stateindex(mdp, s)
    
    @test index == length(states(mdp))
end

# Test POMDPs.isterminal
@testset "Test POMDPs.isterminal" begin
    mdp = create_test_mdp()
    s = State(Cell(5, 1), [false for _ in 1:mdp.size[1]])
    
    @test POMDPs.isterminal(mdp, s) == (s.cell in mdp.terminal_zone)
end

# Test POMDPs.initialstate
@testset "Test POMDPs.initialstate" begin
    mdp = create_test_mdp()
    rng = mdp.rng
    s0 = rand(rng, POMDPs.initialstate(mdp))
    initial_states = POMDPs.initialstate(mdp)
    
    @test typeof(s0) == State   
    # @test length(initial_states[1]) == length(get_target_bounds(mdp, 1))
    
    @test s0.cell[1] == 1
end

# Test POMDPs.actions
@testset "Test POMDPs.actions" begin
    mdp = create_test_mdp()
    actions = POMDPs.actions(mdp)
    
    @test actions == (UP, DOWN, RIGHT)
end

# Test POMDPs.actionindex
@testset "Test POMDPs.actionindex" begin
    mdp = create_test_mdp()
    index_up = POMDPs.actionindex(mdp, UP)
    index_down = POMDPs.actionindex(mdp, DOWN)
    
    @test index_up == 1
    @test index_down == 2
end

# Test POMDPs.transition
@testset "Test POMDPs.transition" begin
    mdp = create_test_mdp()
    s = State(Cell(1, 1), [false for _ in 1:mdp.size[1]])
    transitions = POMDPs.transition(mdp, s, UP)
    
    @test length(transitions) == 3  # Number of possible actions
    @test transitions.probs[1] == 1 - mdp.drift_prob  # Probability of the desired action
end

# Test POMDPs.reward
@testset "Test POMDPs.reward" begin
    mdp = create_test_mdp()
    s = State(Cell(1, 1), [false for _ in 1:mdp.size[1]])
    sp = State(Cell(1, 2), [false for _ in 1:mdp.size[1]])
    
    @test POMDPs.reward(mdp, s, UP, sp) == mdp.reward_offtarget
end

# Test POMDPs.discount
@testset "Test POMDPs.discount" begin
    mdp = create_test_mdp()
    @test POMDPs.discount(mdp) == mdp.discount
end
