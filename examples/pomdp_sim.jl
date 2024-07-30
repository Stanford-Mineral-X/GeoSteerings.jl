using ARDESPOT
using MCTS
using TabularTDLearning

include("src/geosteering.jl")
include("src/mdp.jl")
include("src/utils.jl")

rng = MersenneTwister(1)
p = initialize_pomdp(size=(5,5))
s = rand(rng, initialstate(p))
up = NothingUpdater() #TODO: change with BootstrapFilter or KalmanFilter
b0 = initialize_belief(p)

b = update(up, b0, s, up)
policy = RandomPolicy(p)
a = action(policy, b)
(sp, r, o) = gen(p, s, a, rng)



solver = DESPOTSolver(bounds=IndependentBounds(-5000, 500))
@time policy = solve(solver, p)
hr = HistoryRecorder(max_steps=10, rng=rng)
hist = simulate(hr, p, policy)