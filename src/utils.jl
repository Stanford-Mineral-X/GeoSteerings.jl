
const rgb_darkergray = RGB(0.1, 0.1, 0.1)
const rgb_darkgray = RGB(0.3, 0.3, 0.3)
const rgb_black = RGB(0.0, 0.0, 0.0)
const rgb_lightgray = RGB(0.6, 0.6, 0.6)
const rgb_white = RGB(1.0, 1.0, 1.0)

function cell_ctx(xy, size)
	nx, ny = size
	x, y = xy
	cell_size = 1 / max(nx, ny)
	x_offset = (1 - nx * cell_size) / 2
	y_offset = (1 - ny * cell_size) / 2
	return context(x_offset + (x - 1) * cell_size, y_offset + (ny - y) * cell_size, cell_size, cell_size)
end

function render_cell(mdp::GeoSteeringMDP, cell::Context, pos::Cell)
	if pos in mdp.target_zone
		fill_color = rgb_white
	elseif pos in mdp.shale_zone
		fill_color = rgb_lightgray
	else
		fill_color = rgb_darkgray
	end

	stroke_color = rgb_black
	compose!(cell, Compose.rectangle(), fill(fill_color), Compose.stroke(stroke_color))
end


function render_agent(agent::Union{Nothing, Context}, step::Union{NamedTuple, Dict}, alpha::Float64 = 0.4)
	if haskey(step, :a)
		compose!(agent, Compose.text(0.5, 0.5, aarrow[step[:a]], hcenter, vcenter))
	end
	compose!(agent, Compose.circle(0.5, 0.5, 0.4), fill(RGBA(1.0, 0.65, 0.0, alpha)))
end

function render_next_agent(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, next_agent::Union{Nothing, Context}, step::Union{NamedTuple, Dict})
	if haskey(step, :sp)
		compose!(next_agent, Compose.circle(0.5, 0.5, 0.4), fill(RGBA(1.0, 0.65, 0.0, 0.9)))
	end
end

function render(mdp::GeoSteeringMDP, step::Union{NamedTuple, Dict} = (;))
	nx, ny = mdp.size
	cells = []

	for x in 1:nx, y in 1:ny
		cell = cell_ctx((x, y), mdp.size)
		pos = Cell(x, y)

		render_cell(mdp, cell, pos)

		push!(cells, cell)
	end

	grid = compose(context(), linewidth(0.5mm), cells...)
	outline = compose(context(), linewidth(1mm), Compose.rectangle(), Compose.stroke("black"))

	agent = nothing
	if haskey(step, :s)
		if haskey(step, :sp)
			alpha = 0.4
		else
			alpha = 1.0
		end
		agent = cell_ctx(step[:s].cell, mdp.size)
		render_agent(agent, step, alpha)
	end

	next_agent = nothing
	if haskey(step, :sp)
		next_agent = cell_ctx(step[:sp].cell, mdp.size)
		render_next_agent(mdp, next_agent, step)
	end

	sz = min(w, h)
	return compose(context((w - sz) / 2, (h - sz) / 2, sz, sz), agent, next_agent, grid, outline)
end

function render(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, hist::SimHistory; full::Bool = false)
	render_steps = []
	for step in hist
		if full
			push!(render_steps, render_full(mdp, step))
		else
			push!(render_steps, render(mdp, step))
		end
	end
	return render_steps
end


tofunc(mdp::GeoSteeringMDP, f) = f
tofunc(mdp::GeoSteeringMDP, mat::AbstractMatrix) = s -> mat[s...]
tofunc(mdp::GeoSteeringMDP, v::AbstractVector) = s -> v[stateindex(mdp, s)]

function savefig(obj, (w, h), filename; dpi = 250)
	#try the draw if the PDF is not installed, otherwise catch error    
	#get the filetype from the filename
	_, ext = splitext(filename)
	try
		if ext == ".pdf"
			draw(PDF(filename, w * cm, h * cm), obj)
		elseif ext == ".png"
			draw(PNG(filename, w * cm, h * cm, dpi = dpi), obj)
		elseif ext == ".svg"
			draw(SVG(filename, w * cm, h * cm), obj)
		else
			println("Error saving a figure. Only .pdf, .png, and .svg are supported.")
		end
	catch
		return "Error saving a pdf figure."
	end
end


function create_gif_from_images(; dir = "figs/", gif_name = "SimRollout.gif", num_steps = 10, fps = 2, file_base = "SimRollout")
	image_files = [joinpath(dir, "$file_base$i.png") for i in 1:num_steps]
	images = [load(img_file) for img_file in image_files]

	# Create animation
	anim = @animate for img in images
		plot(heatmap(img),
			legend = false,
			grid = false,          # Disable grid
			ticks = false,         # Disable ticks
			size = (500, 500))     # Size of the plot
	end

	# Save the animation as a GIF
	gif(anim, joinpath(dir, gif_name), fps = fps)
end

function create_side_by_side_gif_from_images(; dir = "figs/", img1_base = "SimRollout", img2_base = "FullRollout", gif_name = "SimRollout.gif", num_steps = 10, fps = 2)
	image_files1 = [joinpath(dir, "$img1_base$i.png") for i in 1:num_steps]
	image_files2 = [joinpath(dir, "$img2_base$i.png") for i in 1:num_steps]

	images1 = [load(img_file) for img_file in image_files1]
	images2 = [load(img_file) for img_file in image_files2]

	#assert that the number of images in both directories is the same
	@assert length(images1) == length(images2)



	# Create animation
	anim = @animate for (i, img1) in enumerate(images1)
		img2 = images2[i]
		#plot side by side
		plot(heatmap(hcat(img1, img2)),
			legend = false,
			grid = false,          # Disable grid
			ticks = false,         # Disable ticks
			#  xlims=(0, 1),        # Set limits
			#  ylims=(0, 1),
			#  xlabel="",           # Remove x-axis label
			#  ylabel="",           # Remove y-axis label
			#  framestyle=:box,     # Box around frame
			size = (1000, 500))     # Size of the plot
	end

	# Save the animation as a GIF
	gif(anim, joinpath(dir, gif_name), fps = fps)
end


function render_observation(pomdp::GeoSteeringPOMDP, cell::Context, pos::Cell, pos_obs::Bool)
	if pos_obs #observed as within formation
		fill_color = rgb_white
	else
		fill_color = rgb_darkgray
	end

	stroke_color = rgb_black
	compose!(cell, Compose.rectangle(), fill(fill_color), Compose.stroke(stroke_color))
	# println("rendering obs at $pos with fill color $fill_color")
end

function render_current_pos(pomdp::GeoSteeringPOMDP, cell::Context, pos::Cell)
	if pos in pomdp.target_zone
		fill_color = rgb_white
	elseif pos in pomdp.shale_zone
		fill_color = rgb_lightgray
	else
		fill_color = rgb_darkgray
	end
	# println("rendering current pos at $pos with fill color $fill_color")

	stroke_color = rgb_black
	compose!(cell, Compose.rectangle(), fill(fill_color), Compose.stroke(stroke_color))
end

function render(pomdp::GeoSteeringPOMDP, step::Union{NamedTuple, Dict})

	agent = nothing
	if haskey(step, :s)

		if haskey(step, :sp)
			pos = step[:sp].cell
			alpha = 0.4
		else
			pos = step[:s].cell
			alpha = 1.0
		end

		#render non surrounding cells
		nx, ny = pomdp.size
		cells = []


		if haskey(step, :o)
			surroundings = step[:o].is_surrounding_target

			for x in 1:nx, y in 1:ny
				cell = cell_ctx((x, y), pomdp.size)
				pos_ = Cell(x, y)
				if pos_ ∉ keys(surroundings) && pos_ !== pos
					# println("rendering non-surrounding cell at $pos_ with fill color $rgb_darkgray")
					compose!(cell, Compose.rectangle(), fill(rgb_darkergray), Compose.stroke(rgb_black))
				else
					if pos_ == pos
						# println("rendering current cell at $pos_ as state")
						render_current_pos(pomdp, cell, pos_)
					else
						if 1 <= pos_[1] <= pomdp.size[1] && 1 <= pos_[2] <= pomdp.size[2]
							# println("rendering surrounding cell at $pos_ as obs")
							obs_ = surroundings[pos_]
							render_observation(pomdp, cell, pos_, obs_)
						end
					end
				end
				push!(cells, cell)
			end
		else
			println("Pass o in step to visualize observations")

		end



		grid = compose(context(), linewidth(0.5mm), cells...)
		outline = compose(context(), linewidth(1mm), Compose.rectangle(), Compose.stroke("black"))

		agent = cell_ctx(step[:s].cell, pomdp.size)
		render_agent(agent, step, alpha)

	end


	next_agent = nothing
	if haskey(step, :sp)

		next_agent = cell_ctx(step[:sp].cell, pomdp.size)
		render_next_agent(pomdp, next_agent, step)
	end

	sz = min(w, h)
	return compose(context((w - sz) / 2, (h - sz) / 2, sz, sz), agent, next_agent, grid, outline)
end


function pomdp2mdp(pomdp::GeoSteeringPOMDP)
	mdp = GeoSteeringMDP(
		size = pomdp.size,
		base_amplitude = pomdp.base_amplitude,
		base_frequency = pomdp.base_frequency,
		amplitude_variation = pomdp.amplitude_variation,
		frequency_variation = pomdp.frequency_variation,
		phase = pomdp.phase,
		vertical_shift = pomdp.vertical_shift,
		target_thickness = pomdp.target_thickness,
		drift_prob = pomdp.drift_prob,
		discount = pomdp.discount,
		reward_target = pomdp.reward_target,
		reward_offtarget = pomdp.reward_offtarget,
		reward_goal = pomdp.reward_goal,
		rng = pomdp.rng,
		target_zone = pomdp.target_zone,
		shale_zone = pomdp.shale_zone,
		terminal_zone = pomdp.terminal_zone,
	)
	return mdp
end

function render_full(pomdp::GeoSteeringPOMDP, step::Union{NamedTuple, Dict})
	mdp_ = pomdp2mdp(pomdp)
	return render(mdp_, step)
end
