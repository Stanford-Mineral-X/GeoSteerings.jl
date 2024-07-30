using ARDESPOT
using MCTS
using TabularTDLearning

include("../src/geosteering.jl")
include("../src/mdp.jl")
include("../src/utils.jl")
include("../src/pomdp.jl")

rng = MersenneTwister(1)
p = initialize_pomdp(size=(5,5))
s = rand(rng, initialstate(p))
up = NothingUpdater() #TODO: change with BootstrapFilter or KalmanFilter
policy = RandomPolicy(p)
hr = HistoryRecorder(max_steps=30, rng=rng)
figs_dir = "figs"
for (policy_num, policy) in enumerate([policy])
    @time hist = simulate(hr, p, policy)
    
    # plot and animate the simulation
    plot_sim_steps = render(p, hist);
    println(length(hist), " steps in the simulation")
    println(length(plot_sim_steps), " plots in the simulation")

    # save the plots as a pdfs
    [savefig(plot_sim_steps[i], p.size, joinpath(figs_dir, "SimRollout$i.pdf")) for i in 1:length(hist)];

    # for animation, we need pngs
    # save the plots as png images 
    [savefig(plot_sim_steps[i], p.size, joinpath(figs_dir, "SimRollout$i.png")) for i in 1:length(hist)];
    #get policy NamedTuple
    policy_name = string(policy)
    create_gif_from_images(dir=figs_dir, gif_name="POMDPPolicy$(policy_num).gif", fps=2, num_steps=length(hist))
end
