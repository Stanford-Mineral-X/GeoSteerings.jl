using Random
using POMDPs
using POMDPTools
using DiscreteValueIteration
using MCTS
using TabularTDLearning
using GeoSteerings

#set the random seed
rng = MersenneTwister(1)

#initialize the MDP with tiny drift_prob
gs = initialize_mdp(
	rng = rng, size = (5, 5),
	base_amplitude = 1.0,
	base_frequency = 1.0,
	target_thickness = 2.0,
	vertical_shift = 3.0,
	drift_prob = 0.001) #no drift_prob

VIsolver = ValueIterationSolver(max_iterations = 50);
@time VIpolicy = solve(VIsolver, gs)

MCTSsolver = MCTSSolver(n_iterations = 500, depth = 50, exploration_constant = 5.0, estimate_value = RolloutEstimator(RandomPolicy(gs)))
@time MCTSpolicy = solve(MCTSsolver, gs)

exploration_policy = EpsGreedyPolicy(gs, 0.1)
qlearning = QLearningSolver(exploration_policy = exploration_policy, learning_rate = 0.05, n_episodes = 2000, max_episode_length = 50, eval_every = 100, n_eval_traj = 100)
@time QLpolicy = solve(qlearning, gs)

# simulate the policy
hr = HistoryRecorder(max_steps = 10, rng = rng)
figs_dir = "figs"
for (policy_num, policy) in enumerate([VIpolicy, QLpolicy, MCTSpolicy])
	@time hist = simulate(hr, gs, policy)

	# plot and animate the simulation
	plot_sim_steps = GeoSteerings.render(gs, hist)
	println(length(hist), " steps in the simulation")
	println(length(plot_sim_steps), " plots in the simulation")

	# save the plots as a pdfs
	[GeoSteerings.savefig(plot_sim_steps[i], gs.size, joinpath(figs_dir, "SimRollout$i.pdf")) for i in 1:length(hist)]

	# for animation, we need pngs
	# save the plots as png images 
	[GeoSteerings.savefig(plot_sim_steps[i], gs.size, joinpath(figs_dir, "SimRollout$i.png")) for i in 1:length(hist)]
	#get policy NamedTuple
	policy_name = string(policy)
	create_gif_from_images(dir = figs_dir, gif_name = "MDPPolicy$(policy_num)_NoDrifting.gif", fps = 2, num_steps = length(hist))
end


#initialize an MDP with some nontrivial drift_prob
gs = initialize_mdp(
	rng = rng, size = (5, 5),
	base_amplitude = 1.0,
	base_frequency = 1.0,
	target_thickness = 2.0,
	vertical_shift = 3.0,
	drift_prob = 0.25) #some drift_prob

VIsolver = ValueIterationSolver(max_iterations = 50);
@time VIpolicy = solve(VIsolver, gs)

MCTSsolver = MCTSSolver(n_iterations = 500, depth = 50, exploration_constant = 5.0, estimate_value = RolloutEstimator(RandomPolicy(gs)))
@time MCTSpolicy = solve(MCTSsolver, gs)

exploration_policy = EpsGreedyPolicy(gs, 0.1)
qlearning = QLearningSolver(exploration_policy = exploration_policy, learning_rate = 0.05, n_episodes = 2000, max_episode_length = 50, eval_every = 100, n_eval_traj = 100)
@time QLpolicy = solve(qlearning, gs)

# simulate the policy
hr = HistoryRecorder(max_steps = 30, rng = rng)
figs_dir = "figs"
for (policy_num, policy) in enumerate([VIpolicy, QLpolicy, MCTSpolicy])
	@time hist = simulate(hr, gs, policy)

	plot_sim_steps = GeoSteerings.render(gs, hist)
	println(length(hist), " steps in the simulation")
	println(length(plot_sim_steps), " plots in the simulation")

	[GeoSteerings.savefig(plot_sim_steps[i], gs.size, joinpath(figs_dir, "SimRollout$i.pdf")) for i in 1:length(hist)]

	[GeoSteerings.savefig(plot_sim_steps[i], gs.size, joinpath(figs_dir, "SimRollout$i.png")) for i in 1:length(hist)]

	policy_name = string(policy)
	create_gif_from_images(dir = figs_dir, gif_name = "MDPPolicy$(policy_num)_WithDrifting.gif", fps = 2, num_steps = length(hist))
end
