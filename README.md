# GeoSteerings.jl

[![CI](https://github.com/mansurarief/GeoSteerings.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mansurarief/GeoSteerings.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/mansurarief/GeoSteerings.jl/graph/badge.svg?token=4PNXS83ILY)](https://codecov.io/gh/mansurarief/GeoSteerings.jl)

This repository contains implementations of GeoSteeringMDP and GeoSteeringPOMDP problems in Julia. *GeoSteering* is a process used in the subsurface exploration industry to guide the wellbore of a drilling operation in real-time, ensuring it stays within a predefined target. This complex problem involves making sequential decisions based on uncertain and incomplete information, which can be effectively modeled using Markov Decision Processes (MDP) and Partially Observable Markov Decision Processes (POMDP).

## Notebooks

Notebook examples are available in [GeoSteeringMDP](https://github.com/mansurarief/GeoSteerings.jl/blob/main/notebooks/GeoSteeringMDP.ipynb) and [GeoSteeringPOMDP](https://github.com/mansurarief/GeoSteerings.jl/blob/main/notebooks/GeoSteeringPOMDP.ipynb).


## MDP Example

![GeoSteeringProblem](./figs/MDPPolicy1_NoDrifting.gif)

```julia
using Random, POMDPs, POMDPTools
using GeoSteerings

#set the random seed
rng = MersenneTwister(1)

#initialize the MDP with tiny drift_prob
gs = initialize_mdp(
    rng=rng, size=(5, 5), base_amplitude=1.0, 
    base_frequency=1.0, target_thickness=2.0, 
    vertical_shift=3.0, drift_prob=0.001) 

# simulate 
hr = HistoryRecorder(max_steps=10, rng=rng)
@time hist = simulate(hr, gs, RandomPolicy(gs))

# plot and animate the simulation
plot_sim_steps = render(gs, hist);

[savefig(plot_sim_steps[i], gs.size, joinpath("figs", "SimRollout$i.pdf")) for i in 1:length(hist)];
[savefig(plot_sim_steps[i], gs.size, joinpath("figs", "SimRollout$i.png")) for i in 1:length(hist)];

create_gif_from_images(dir=figs_dir, gif_name="MDPPolicy1_NoDrifting.gif", fps=2, num_steps=length(hist))

```

## POMDP Example

![GeoSteeringPOMDPProblem](./figs/POMDPPolicy2_WithDrifting.gif)

```julia
using POMDPs, POMDPTools, GeoSteerings, ParticleFilters, POMCPOW

# set the random seed
rng = MersenneTwister(1)

gs = initialize_pomdp(
    rng=rng, size=(5, 5), base_amplitude=1.0, 
    base_frequency=1.0, target_thickness=2.0, 
    vertical_shift=2.0, drift_prob=0.1) #adding some drift_prob

# initialization
up = BootstrapFilter(gs, 100, rng)
b0 = ParticleCollection(support(initialize_belief(gs)))
s0 = rand(rng, initialstate(gs))

# test a solver
solver = POMCPOWSolver(tree_queries=50,criterion=MaxUCB(100.0))
policy = solve(solver, gs)

# simulate
hr = HistoryRecorder(max_steps=30, rng=rng)
@time hist_ = simulate(hr, gs, policy, up, b0)

# plot and animate 
plot_sim_steps = render(gs, hist_);
plot_sim_full = render(gs, hist_, full=true);

[savefig(plot_sim_steps[i], gs.size, joinpath("figs", "$base_img_sim$i.png")) for i in 1:length(hist_)];
[savefig(plot_sim_full[i], gs.size, joinpath("figs", "$base_img_full$i.png")) for i in 1:length(hist_)];

create_side_by_side_gif_from_images(dir="figs", img1_base="SimRollout", img2_base="FullRollout", 
    gif_name="POMDPPolicy$(policy_num)_WithDrifting.gif", fps=1, num_steps=length(hist_))

```

## Maintainer

Maintained by Mansur M. Arief (mansur.arief@stanford.edu). Please submit a pull request or reach out if you have any suggestions.