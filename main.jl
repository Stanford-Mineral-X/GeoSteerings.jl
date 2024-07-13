using MCTS
using GeoSteeringPOMDPs
using POMDPs
using POMDPSimulators
using POMDPTools


gs = GeoSteering()

solver = MCTSSolver(n_iterations=10000, depth=100, exploration_constant=50.0)  
policy = solve(solver, gs)

hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1))
hist = simulate(hr, gs, policy)

render(gs, (s=[1,5],), policy=policy)