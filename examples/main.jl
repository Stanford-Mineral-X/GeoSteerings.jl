using POMDPs
using GeoSteerings
using Random

rng = MersenneTwister(1)
gs = initialize_mdp(
    rng=rng, size=(5, 5), 
    base_amplitude=1.0, 
    base_frequency=1.0,
    target_thickness=2.0, 
    vertical_shift=3.0,
    drift_prob=0.0001) #no drift_prob
println(length(states(gs)))  

gs = initialize_mdp(
    rng=rng, size=(4, 4), 
    base_amplitude=1.0, 
    base_frequency=1.0,
    target_thickness=2.0, 
    vertical_shift=3.0,
    drift_prob=0.0001) #no drift_prob
println(length(states(gs))) 

gs = initialize_mdp(
    rng=rng, size=(3, 3), 
    base_amplitude=1.0, 
    base_frequency=1.0,
    target_thickness=1.0, 
    vertical_shift=1.0,
    drift_prob=0.0001) #no drift_prob    
println(length(states(gs)))

# one simulation step example
p0 = initialstate(gs) # initial state distribution
s = rand(rng, p0) # initial state s ~ p0
a = rand(rng, actions(gs)) # random action a0
p = transition(gs, s, a) # next state distribution p
sp = rand(rng, p) # next state s ~ p
r = reward(gs, s, a, sp) # reward r


#plot the initial state
plt = GeoSteerings.render(gs, (s=s,a=a,sp=sp))
GeoSteerings.savefig(plt, gs.size, "figs/MDP-1step-simulation.pdf")


