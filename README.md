# GeoSteerings.jl

This repository contains implementations of GeoSteeringMDP and GeoSteeringPOMDP problems in Julia. *GeoSteering* is a process used in the subsurface exploration industry to guide the wellbore of a drilling operation in real-time, ensuring it stays within a predefined target. This complex problem involves making sequential decisions based on uncertain and incomplete information, which can be effectively modeled using Markov Decision Processes (MDP) and Partially Observable Markov Decision Processes (POMDP).

## MDP Example

![GeoSteeringProblem](./figs/GeoSteeringProblem.png)

```julia
using GeoSteerings
using DiscreteValueIteration
using POMDPs
using POMDPSimulators
using Random

# Initialize the GeoSteering problem
gs = GeoSteering()

# Define the offline policy solver
solver = ValueIterationSolver(max_iterations=1000)
policy = solve(solver, gs)

# Simulate the policy
hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1))
hist = simulate(hr, gs, policy)

# Visualize the policy
plot2 = GeoSteerings.render(gs, (s=[1,5],), policy=policy)
savefig(plot2, (10, 10), "figs/GeoSteeringPolicyExample.pdf")
```

This MDP example is also available in the [GeoSteeringMDP notebook](./notebooks/GeoSteeringMDP.ipynb) or [GeoSteeringMDP.jl file](GeoSteeringMDP_example.jl)

## POMDP Example

To be added (TBA).