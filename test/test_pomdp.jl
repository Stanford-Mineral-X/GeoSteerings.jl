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
include("../src/pomdp.jl")

# Define a helper function to create a test GeoSteeringMDP
function create_test_pomdp()
    return initialize_pomdp(size=(3, 3), base_amplitude=1.0, base_frequency=1.0,
                          amplitude_variation=0.01, frequency_variation=0.01, phase=0.2,
                          vertical_shift=1.0, target_thickness=1.0, drift_prob=0.2, obs_tol=1,
                          discount=0.9, reward_target=50.0, reward_offtarget=-50.0, reward_goal=1000.0,
                          rng=Random.GLOBAL_RNG)
end

pomdp = create_test_pomdp()

@testset "Test POMDPs.initialize_belief" begin
    init_belief = POMDPs.initialize_belief(pomdp)
    @test typeof(init_belief) == SparseCat{Vector{State}, Vector{Float64}}
end

𝒪 = POMDPs.observations(pomdp);

# Test POMDPs.observations
@testset "Test POMDPs.observations" begin
    obs = 𝒪[1]
    @test obs.is_surrounding_target[Cell(0,0)] == false

end

obs = 𝒪[1]


@testset "Base.EqualObs" begin
    obs = 𝒪[1]
    @test obs == obs
end
obs_dist = POMDPs.observation(pomdp, State(Cell(1, 1), get_surrounding_status(pomdp, Cell(1, 1))))


# Test POMDPs.observation
@testset "Test POMDPs.obsindex" begin
    obs_dist = POMDPs.observation(pomdp, State(Cell(1, 1), get_surrounding_status(pomdp, Cell(1, 1))))
    @test typeof(obs_dist) == SparseCat{Vector{Any}, Vector{Float64}}

end

# Test POMDPs.gen
@testset "Test POMDPs.get" begin    
    s = State(Cell(1, 1), get_surrounding_status(pomdp, Cell(1, 1)))
    a = UP
    rng = Random.GLOBAL_RNG
    sp, r, o = POMDPs.gen(pomdp, s, a, rng)
    
    @test isa(sp, State)
    @test isa(r, Float64)
    @test isa(o, Any)
end