#solving using various solvers
using DiscreteValueIteration
using MCTS
using TabularTDLearning
# using POMDPSimulators

include("../src/geosteering.jl")
include("../src/mdp.jl")
include("../src/utils.jl")

rng = MersenneTwister(1)
gs = initialize_mdp(
    rng=rng, size=(5, 5), 
    base_amplitude=1.0, 
    base_frequency=1.0,
    target_thickness=2.0, 
    vertical_shift=3.0,
    drift_prob=0.0001) #no drift_prob

# VIsolver = ValueIterationSolver(max_iterations=10);
# @time VIpolicy = solve(VIsolver, gs)

MCTSsolver = MCTSSolver(n_iterations=500, depth=50, exploration_constant=5.0, estimate_value=RolloutEstimator(RandomPolicy(gs)))
@time MCTSpolicy = solve(MCTSsolver, gs)

exploration_policy = EpsGreedyPolicy(gs, 0.01)
qlearning = QLearningSolver(exploration_policy=exploration_policy, learning_rate=0.1, n_episodes=1000, max_episode_length=50, eval_every=50, n_eval_traj=100)
@time QLpolicy = solve(qlearning, gs)

# simulate the policy
hr = HistoryRecorder(max_steps=10, rng=rng)
figs_dir = "../figs"
for (policy_num, policy) in enumerate([QLpolicy, MCTSpolicy])
    @time hist = simulate(hr, gs, policy)
    
    # plot and animate the simulation
    plot_sim_steps = render(gs, hist);
    println(length(hist), " steps in the simulation")
    println(length(plot_sim_steps), " plots in the simulation")

    # save the plots as a pdfs
    [savefig(plot_sim_steps[i], gs.size, joinpath(figs_dir, "SimRollout$i.pdf")) for i in 1:length(hist)];

    # for animation, we need pngs
    # save the plots as png images 
    [savefig(plot_sim_steps[i], gs.size, joinpath(figs_dir, "SimRollout$i.png")) for i in 1:length(hist)];
    #get policy NamedTuple
    policy_name = string(policy)
    create_gif_from_images(dir=figs_dir, gif_name="SimPolicy$(policy_num).gif", fps=2, num_steps=length(hist))
end