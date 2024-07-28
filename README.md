# GeoSteerings.jl
[![CI](https://github.com/mansurarief/GeoSteerings.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mansurarief/GeoSteerings.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/mansurarief/GeoSteerings.jl/graph/badge.svg?token=4PNXS83ILY)](https://codecov.io/gh/mansurarief/GeoSteerings.jl)


This repository contains implementations of GeoSteeringMDP and GeoSteeringPOMDP problems in Julia. *GeoSteering* is a process used in the subsurface exploration industry to guide the wellbore of a drilling operation in real-time, ensuring it stays within a predefined target. This complex problem involves making sequential decisions based on uncertain and incomplete information, which can be effectively modeled using Markov Decision Processes (MDP) and Partially Observable Markov Decision Processes (POMDP).

## MDP Example

![GeoSteeringProblem](./figs/SimPolicyVI.gif)

```julia
include("../src/geosteering.jl")
include("../src/mdp.jl")
include("../src/utils.jl")

rng = MersenneTwister(1)
gs = initialize_mdp(
    rng=rng, size=(10, 10), 
    base_amplitude=2.0, 
    target_thickness=3.0, 
    vertical_shift=5.0,
    drift_prob=0.01)

# one simulation step example
p0 = initialstate(gs) # initial state distribution
s = rand(rng, p0) # initial state s ~ p0
a = rand(rng, actions(gs)) # random action a0
p = transition(gs, s, a) # next state distribution p
sp = rand(rng, p) # next state s ~ p
r = reward(gs, s, a, sp) # reward r


#plot the initial state
plt = render(gs, (s=s,a=a,sp=sp));
savefig(plt, gs.size, "../figs/MDP-1step-simulation.pdf")
ing()

# Define the offline policy solver
solver = ValueIterationSolver(max_iterations=1000)
policy = solve(solver, gs)

# Simulate the policy
hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1))
hist = simulate(hr, gs, policy)

# Visualize the simulation result
plot2 = ender(gs, hist)
savefig(plot2, gs.size, "../figs/GeoSteeringPolicyExample.pdf")
```

This MDP example is also available in the [GeoSteeringMDP notebook](../notebooks/GeoSteeringMDP.ipynb) or [example.jl file](../examples/example.jl)

## POMDP Example

To be added (TBA).
