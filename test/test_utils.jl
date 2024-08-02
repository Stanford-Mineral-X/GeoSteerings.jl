using Test
using Random
using POMDPs
using POMDPTools
using POMDPModels
using Parameters
using StaticArrays
using Distributions
using Colors
using IterTools

# Include the code you want to test here (or use include to bring it in)
include("../src/geosteering.jl")
include("../src/mdp.jl")
include("../src/pomdp.jl")
include("../src/utils.jl")

# Define a helper function to create a test GeoSteeringMDP
function create_test_pomdp()
	return initialize_pomdp(size = (3, 3), base_amplitude = 1.0, base_frequency = 1.0,
		amplitude_variation = 0.01, frequency_variation = 0.01, phase = 0.2,
		vertical_shift = 1.0, target_thickness = 1.0, drift_prob = 0.2, obs_tol = 1,
		discount = 0.9, reward_target = 50.0, reward_offtarget = -50.0, reward_goal = 1000.0,
		rng = Random.GLOBAL_RNG)
end

function create_test_mdp()
	return initialize_mdp(size = (3, 3), base_amplitude = 1.0, base_frequency = 1.0,
		amplitude_variation = 0.01, frequency_variation = 0.01, phase = 0.2,
		vertical_shift = 1.0, target_thickness = 1.0, drift_prob = 0.2,
		discount = 0.9, reward_target = 50.0, reward_offtarget = -50.0, reward_goal = 1000.0,
		rng = Random.GLOBAL_RNG)
end

mdp = create_test_mdp()
pomdp = create_test_pomdp()
s = State(Cell(1, 1), get_surrounding_status(mdp, Cell(1, 1)))
ctx = cell_ctx(Cell(1, 1), mdp.size);
step = (s = s, a = RIGHT, sp = s)

@testset "Utils constants" begin
	@test rgb_darkergray == RGB(0.1, 0.1, 0.1)
	@test rgb_darkgray == RGB(0.3, 0.3, 0.3)
	@test rgb_black == RGB(0.0, 0.0, 0.0)
	@test rgb_lightgray == RGB(0.6, 0.6, 0.6)
	@test rgb_white == RGB(1.0, 1.0, 1.0)
end


@testset "Utils cell_ctx" begin
	size = mdp.size
	@test typeof(ctx) == Context
end


@testset "Utils render_cell" begin
	pos = [mdp.target_zone...][1]
	case_target = render_cell(mdp, cell_ctx(pos, mdp.size), pos)

	pos = [mdp.shale_zone...][1]
	case_shale = render_cell(mdp, cell_ctx(pos, mdp.size), pos)

	pos = Cell(mdp.size[1], mdp.size[2])
	case_else = render_cell(mdp, cell_ctx(pos, mdp.size), pos)

	@test typeof(case_target) == Context
	@test typeof(case_shale) == Context
	@test typeof(case_else) == Context
end


@testset "Utils render_agent" begin
	step_with_action = (s = s, a = RIGHT)
	step_without_action = (s = s,)

	rendered_with_action = render_agent(ctx, step_with_action)
	rendered_without_action = render_agent(ctx, step_without_action)

	@test typeof(rendered_with_action) == Context
	@test typeof(rendered_without_action) == Context
end

@testset "Utils render_next_agent" begin
	sp = deepcopy(s)
	step_with_sp = (s = s, a = RIGHT, sp = sp)
	step_without_sp = (s = s, a = RIGHT)

	rendered_with_sp = render_next_agent(mdp, ctx, step_with_sp)
	rendered_without_sp = render_next_agent(mdp, ctx, step_without_sp)

	@test typeof(rendered_with_sp) == Context
	@test isnothing(rendered_without_sp)
end


@testset "Utils render MDP" begin
	rendered = render(mdp, step)
	@test typeof(rendered) == Context

	empty_step = (;)
	rendered_empty = render(mdp, empty_step)
	@test typeof(rendered_empty) == Context
end

@testset "Utils render POMDP" begin
	step_o = (s = s, a = RIGHT, sp = s, o = Observation(get_surrounding_status(mdp, s.cell)))
	rendered = render(pomdp, step_o)
	@test typeof(rendered) == Context

	rendered_without_o = render(pomdp, step)
	@test typeof(rendered_without_o) == Context
end


@testset "Utils render_observation" begin
	pos = Cell(1, 1)
	rendered_true = render_observation(pomdp, ctx, pos, true)
	rendered_false = render_observation(pomdp, ctx, pos, false)

	@test typeof(rendered_true) == Context
	@test typeof(rendered_false) == Context
end

@testset "Utils render_current_pos" begin
	pos_target = [pomdp.target_zone...][1]
	pos_shale = [pomdp.shale_zone...][1]
	pos_other = Cell(pomdp.size[1], pomdp.size[2])

	rendered_target = render_current_pos(pomdp, ctx, pos_target)
	rendered_shale = render_current_pos(pomdp, ctx, pos_shale)
	rendered_other = render_current_pos(pomdp, ctx, pos_other)

	@test typeof(rendered_target) == Context
	@test typeof(rendered_shale) == Context
	@test typeof(rendered_other) == Context
end

@testset "Utils pomdp2mdp" begin
	converted_mdp = pomdp2mdp(pomdp)
	@test typeof(converted_mdp) == GeoSteeringMDP
	@test converted_mdp.size == pomdp.size
	@test converted_mdp.base_amplitude == pomdp.base_amplitude
	@test converted_mdp.base_frequency == pomdp.base_frequency
	@test converted_mdp.amplitude_variation == pomdp.amplitude_variation
	@test converted_mdp.frequency_variation == pomdp.frequency_variation
	@test converted_mdp.phase == pomdp.phase
	@test converted_mdp.vertical_shift == pomdp.vertical_shift
	@test converted_mdp.target_thickness == pomdp.target_thickness
	@test converted_mdp.drift_prob == pomdp.drift_prob
	@test converted_mdp.discount == pomdp.discount
	@test converted_mdp.reward_target == pomdp.reward_target
	@test converted_mdp.reward_offtarget == pomdp.reward_offtarget
	@test converted_mdp.reward_goal == pomdp.reward_goal
	@test converted_mdp.target_zone == pomdp.target_zone
	@test converted_mdp.shale_zone == pomdp.shale_zone
	@test converted_mdp.terminal_zone == pomdp.terminal_zone
end

@testset "Utils render_full" begin
	rendered = render_full(pomdp, step)
	@test typeof(rendered) == Context
end

@testset "Save figs" begin
	rendered = render_full(pomdp, step)


	savefig(rendered, pomdp.size, "test.pdf")
	savefig(rendered, pomdp.size, "test.png")
	savefig(rendered, pomdp.size, "test.svg")

	@test isfile("test.pdf")
	@test isfile("test.png")
	@test isfile("test.svg")
end


@testset "Save gif" begin
	steps = []
	num_steps = 2
	for i in 1:num_steps
		push!(steps, (s = s, a = RIGHT, sp = s, o = Observation(get_surrounding_status(mdp, s.cell))))
	end

	rendered = [render_full(pomdp, step) for step in steps]

	for (i, fig) in enumerate(rendered)
		savefig(fig, pomdp.size, "test$i.png")
	end

	create_gif_from_images(dir = "", gif_name = "test.gif", file_base = "test", num_steps = num_steps, fps = 1)

	@test isfile("test.gif")

end

@testset "Save gif side_by_side" begin
	steps = []
	num_steps = 2
	for i in 1:num_steps
		push!(steps, (s = s, a = RIGHT, sp = s, o = Observation(get_surrounding_status(mdp, s.cell))))
	end

	rendered = [render_full(pomdp, step) for step in steps]

	for (i, fig) in enumerate(rendered)
		savefig(fig, pomdp.size, "test$i.png")
	end

	create_side_by_side_gif_from_images(dir = "", img1_base = "test", img2_base = "test",
		gif_name = "test_side.gif", fps = 1, num_steps = num_steps)

	@test isfile("test_side.gif")

end

#clean up the files
rm("test.pdf", force = true)
rm("test.png", force = true)
rm("test.svg", force = true)
rm("test.gif", force = true)
rm("test1.png", force = true)
rm("test2.png", force = true)
rm("test_side.gif", force = true)
