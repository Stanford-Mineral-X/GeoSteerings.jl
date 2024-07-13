using GeoSteerings
using DiscreteValueIteration
using POMDPs
using POMDPSimulators
using Random

gs = GeoSteering()

plot1 = GeoSteerings.render(gs, (s=[1,5],))
savefig(plot1, (10, 10), "figs/GeoSteeringProblem.pdf")

#offline solver
solver = ValueIterationSolver(max_iterations=1000);
policy = solve(solver, gs)

hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1))
hist = simulate(hr, gs, policy)

plot2 = GeoSteerings.render(gs, (s=[1,5],), policy=policy)
savefig(plot2, (10, 10), "figs/GeoSteeringPolicyExample.pdf")
