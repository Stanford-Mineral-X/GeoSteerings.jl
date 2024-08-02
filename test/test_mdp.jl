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
    return initialize_mdp(size=(3, 3), base_amplitude=1.0, base_frequency=1.0,
                          amplitude_variation=0.01, frequency_variation=0.01, phase=0.2,
                          vertical_shift=1.0, target_thickness=1.0, drift_prob=0.2, 
                          discount=0.9, reward_target=50.0, reward_offtarget=-50.0, reward_goal=1000.0,
                          rng=Random.GLOBAL_RNG)
end

mdp = create_test_mdp()

@testset "Test POMDPs.states" begin
    states = POMDPs.states(mdp)
    
    num_cells = prod(mdp.size) + 1
    num_surroundings = prod(mdp.size)
    @test length(states) == num_cells * num_surroundings
    
    # Check some sample states
    @test states[1].cell == Cell(1, 1)
    @test states[end].cell == Cell(-1, -1)
end

cell = Cell(1, 1)
surrounding = get_surrounding_status(mdp, cell)
s = State(cell, surrounding)    


# Test POMDPs.stateindex
@testset "Test POMDPs.stateindex" begin
    
    cell = Cell(1, 1)
    surrounding = get_surrounding_status(mdp, cell)
    s = State(cell, surrounding)    
    @test POMDPs.stateindex(mdp, s) == 1  

    cell = Cell(0, 0)
    surrounding = get_surrounding_status(mdp, Cell(1, 1))
    s = State(cell, surrounding)    
    @test isnothing(POMDPs.stateindex(mdp, s))

end


# Test POMDPs.isterminal
@testset "Test POMDPs.isterminal" begin
    term_state = [mdp.terminal_zone...][1]
    surr = get_surrounding_status(mdp, term_state)
    s = State(term_state, surr)
    
    @test POMDPs.isterminal(mdp, s) == true
end

# Test POMDPs.initialstate
@testset "Test POMDPs.initialstate" begin    
    initial_states = POMDPs.initialstate(mdp)
    s0 = rand(mdp.rng, initial_states)
        
    @test s0.cell[1] == 1
    @test s0 in support(initial_states)
end

# Test POMDPs.actions
@testset "Test POMDPs.actions" begin
    actions = POMDPs.actions(mdp)
    
    @test actions == (UP, DOWN, RIGHT)
end

# Test POMDPs.actionindex
@testset "Test POMDPs.actionindex" begin
    index_up = POMDPs.actionindex(mdp, UP)
    index_down = POMDPs.actionindex(mdp, DOWN)
    
    @test index_up == 1
    @test index_down == 2
end

# Test POMDPs.transition
@testset "Test POMDPs.transition" begin
    cell = Cell(1, 1)
    surrounding = get_surrounding_status(mdp, cell)
    s = State(cell, surrounding)

    transitions = POMDPs.transition(mdp, s, UP)
    
    @test length(transitions) == 3  # Number of possible actions
    @test transitions.probs[1] == 1 - mdp.drift_prob  # Probability of the desired action
end

# Test POMDPs.reward
@testset "Test POMDPs.reward" begin
    s = State(Cell(1, 1), get_surrounding_status(mdp, Cell(1, 1)))
    sp = State(Cell(1, 2), get_surrounding_status(mdp, Cell(1, 2)))
    sp_off = State(Cell(1, 3), get_surrounding_status(mdp, Cell(1, 3)))
    sp_goal = State([mdp.terminal_zone...][1], get_surrounding_status(mdp, [mdp.terminal_zone...][1]))

    @test POMDPs.reward(mdp, s, RIGHT, sp) == mdp.reward_target
    @test POMDPs.reward(mdp, sp, RIGHT, sp_goal) == mdp.reward_goal
    @test POMDPs.reward(mdp, sp, DOWN, sp_off) == mdp.reward_offtarget 
end


#Test POMDPs.gen
@testset "Test POMDPs.gen" begin
    s = State(Cell(1, 1), get_surrounding_status(mdp, Cell(1, 1)))
    a = RIGHT
    sp, r = POMDPs.gen(mdp, s, a, mdp.rng)
    @test typeof(sp) == State
    @test typeof(r) == Float64
end


# Test POMDPs.discount
@testset "Test POMDPs.discount" begin
    mdp = create_test_mdp()
    @test POMDPs.discount(mdp) == mdp.discount
end
