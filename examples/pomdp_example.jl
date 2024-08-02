using ParticleFilters
using POMCPOW
using MCTS
using Random
using POMDPs
using POMDPTools
using GeoSteerings


rng = MersenneTwister(1)
gs = initialize_pomdp(
	rng = rng, size = (5, 5),
	base_amplitude = 1.0,
	base_frequency = 1.0,
	target_thickness = 2.0,
	vertical_shift = 2.0,
	drift_prob = 0.1) #some drift_prob

up = BootstrapFilter(gs, 100, rng)
b0 = ParticleCollection(support(initialize_belief(gs)))
s0 = rand(rng, initialstate(gs))

rpolicy = RandomPolicy(gs)

# a = action(rpolicy, b0)
# sp, r, o = gen(gs, s0, a, rng)
# b = update(up, b0, a, o)
# render(gs, (s=s0, sp=sp, o=o))    

solver = POMCPOWSolver(tree_queries = 50, criterion = MaxUCB(100.0))
policy = solve(solver, gs)

mdp = GenerativeBeliefMDP(gs, up)
msolver = DPWSolver(n_iterations = 50, depth = gs.size[1], estimate_value = RolloutEstimator(rpolicy, max_depth = gs.size[1]))
mpolicy = solve(msolver, mdp)


hr = HistoryRecorder(max_steps = 30, rng = rng)
@time hist = simulate(hr, gs, policy, up, b0)
@time mhist = simulate(hr, gs, mpolicy, up, b0)
@time rhist = simulate(hr, gs, rpolicy, up, b0)

println(""""
	Cumulative reward for 1 simulation run:
			Random: $(discounted_reward(rhist))
			POMCPOW: $(discounted_reward(hist))
			DPW: $(discounted_reward(mhist))
""")


hr = HistoryRecorder(max_steps = 30, rng = rng)
figs_dir = "figs"
for (policy_num, policy) in enumerate([policy, mpolicy, rpolicy])
	@time hist_ = simulate(hr, gs, policy, up, b0)

	# plot and animate the simulation
	plot_sim_steps = GeoSteerings.render(gs, hist_)
	plot_sim_full = GeoSteerings.render(gs, hist_, full = true)

	println(length(hist_), " steps in the simulation")
	println(length(plot_sim_steps), " plots in the simulation")
	println(length(plot_sim_full), " plots in the full rendering")

	# # save the plots as a pdfs
	# [savefig(plot_sim_steps[i], gs.size, joinpath(figs_dir, "SimRollout$i.pdf")) for i in 1:length(hist)];

	# for animation, we need pngs
	# save the plots as png images 
	base_img_sim = "SimRollout"
	base_img_full = "FullRollout"
	[GeoSteerings.savefig(plot_sim_steps[i], gs.size, joinpath(figs_dir, "$base_img_sim$i.png")) for i in 1:length(hist_)]
	[GeoSteerings.savefig(plot_sim_full[i], gs.size, joinpath(figs_dir, "$base_img_full$i.png")) for i in 1:length(hist_)]

	#get policy NamedTuple
	policy_name = string(policy)
	create_side_by_side_gif_from_images(
		dir = figs_dir,
		img1_base = base_img_sim,
		img2_base = base_img_full,
		gif_name = "POMDPPolicy$(policy_num)_WithDrifting.gif",
		fps = 1, num_steps = length(hist_),
	)
end
