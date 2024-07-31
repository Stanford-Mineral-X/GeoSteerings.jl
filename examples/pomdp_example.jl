using ParticleFilters
using POMCPOW
using MCTS

include("../src/geosteering.jl")
include("../src/mdp.jl")
include("../src/utils.jl")
include("../src/pomdp.jl")

rng = MersenneTwister(1)
gs = initialize_pomdp(
    rng=rng, size=(4, 4), 
    base_amplitude=1.0, 
    base_frequency=1.0,
    target_thickness=2.0, 
    vertical_shift=2.0,
    drift_prob=0.0001) #no drift_prob

up = BootstrapFilter(gs, 300, rng)
b0 = ParticleCollection(support(initialize_belief(gs)))

rpolicy = RandomPolicy(gs)

solver = POMCPOWSolver(tree_queries=1000,criterion=MaxUCB(100.0))
policy = solve(solver, gs)

mdp = GenerativeBeliefMDP(gs, up)    
msolver = DPWSolver(n_iterations=10, depth=gs.size[1], estimate_value=RolloutEstimator(rpolicy, max_depth=gs.size[1]))
mpolicy = solve(msolver, mdp)


hr = HistoryRecorder(max_steps=10, rng=rng)
@time hist = simulate(hr, gs, policy, up, b0)
@time mhist = simulate(hr, gs, mpolicy, up, b0)
@time rhist = simulate(hr, gs, rpolicy, up, b0)

println(""""
    Cumulative reward for 1 simulation run:
            Random: $(discounted_reward(rhist))
            POMCPOW: $(discounted_reward(hist))
            DPW: $(discounted_reward(mhist))
""")

