

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

function render_cell(mdp::Union{GeoSteeringPOMDP, GeoSteeringMDP}, cell::Context, pos::Cell)
    if pos in mdp.target_zone
        compose!(cell, Compose.rectangle(), fill(rgb_white), Compose.stroke(rgb_black))
    elseif pos in mdp.shale_zone
        compose!(cell, Compose.rectangle(), fill(rgb_lightgray), Compose.stroke(rgb_black))
    else
        compose!(cell, Compose.rectangle(), fill(rgb_darkgray), Compose.stroke(rgb_black))
    end
end


function render_agent(agent::Union{Nothing, Context}, step::Union{NamedTuple,Dict}, alpha::Float64=0.4)
    if haskey(step, :a)
        compose!(agent, Compose.text(0.5, 0.5, aarrow[step[:a]], hcenter, vcenter))                
    end
    compose!(agent, Compose.circle(0.5, 0.5, 0.4), fill(RGBA(1.0, 0.65, 0.0, alpha)))
end

function render_next_agent(mdp::Union{GeoSteeringPOMDP, GeoSteeringMDP}, next_agent::Union{Nothing, Context}, step::Union{NamedTuple,Dict})
    if haskey(step, :sp)
        compose!(next_agent, Compose.circle(0.5, 0.5, 0.4), fill(RGBA(1.0, 0.65, 0.0, 0.9)))
    end
end

function render(mdp::Union{GeoSteeringPOMDP, GeoSteeringMDP}, step::Union{NamedTuple,Dict}=(;))
    nx, ny = mdp.size
    cells = []
    
    for x in 1:nx, y in 1:ny
        cell = cell_ctx((x, y), mdp.size)
        pos = Cell(x, y)
        
        render_cell( mdp, cell, pos)
        
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

function render(mdp::Union{GeoSteeringPOMDP, GeoSteeringMDP}, hist::SimHistory)
    render_steps = []
    for step in hist
        push!(render_steps, render(mdp, step))
    end
    return render_steps
end


tofunc(mdp::Union{GeoSteeringPOMDP, GeoSteeringMDP}, f) = f
tofunc(mdp::Union{GeoSteeringPOMDP, GeoSteeringMDP}, mat::AbstractMatrix) = s -> mat[s...]
tofunc(mdp::Union{GeoSteeringPOMDP, GeoSteeringMDP}, v::AbstractVector) = s -> v[stateindex(mdp, s)]

function savefig(obj, (w, h), filename; dpi=250)
    #try the draw if the PDF is not installed, otherwise catch error    
    #get the filetype from the filename
    _, ext = splitext(filename)
    try 
        if ext == ".pdf"
            draw(PDF(filename, w*cm, h*cm), obj)
        elseif ext == ".png"
            draw(PNG(filename, w*cm, h*cm, dpi=dpi), obj)
        elseif ext == ".svg"
            draw(SVG(filename, w*cm, h*cm), obj)
        else
            println("Error saving a figure. Only .pdf, .png, and .svg are supported.")
        end
    catch
        return "Error saving a pdf figure."
    end
end


function create_gif_from_images(;dir="figs/", gif_name="SimRollout.gif", num_steps=10, fps=2)
    image_files = [joinpath(dir, "SimRollout$i.png") for i in 1:num_steps]
    images = [load(img_file) for img_file in image_files]

    # Create animation
    anim = @animate for img in images
        plot(heatmap(img), 
             legend=false, 
             grid=false,          # Disable grid
             ticks=false,         # Disable ticks
            #  xlims=(0, 1),        # Set limits
            #  ylims=(0, 1),
            #  xlabel="",           # Remove x-axis label
            #  ylabel="",           # Remove y-axis label
            #  framestyle=:box,     # Box around frame
             size=(500, 500))     # Size of the plot
    end 

    # Save the animation as a GIF
    gif(anim, joinpath(dir, gif_name), fps=fps)
end