function render(mdp::GeoSteeringMDP, step::Union{NamedTuple,Dict}=(;);
    color=s -> reward(mdp, s),
    policy::Union{Policy,Nothing}=nothing,
    colormin::Float64=-10.0, colormax::Float64=10.0
)

    color = tofunc(mdp, color)

    nx, ny = mdp.size
    cells = []
    for x in 1:nx, y in 1:ny
        cell = cell_ctx((x, y), mdp.size)
        if GWPos(x, y) in mdp.blocks  # Check if the cell is a block
            compose!(cell, Compose.rectangle(), fill("black"), Compose.stroke("gray"))  # Render as a grey block
        else
            if policy !== nothing && !(GWPos(x, y) in mdp.terminate_from)  # Check if the cell is not a goal
                a = action(policy, GWPos(x,y))
                txt = compose(context(), Compose.text(0.5, 0.5, aarrow[a], hcenter, vcenter), Compose.stroke("black"), fill("black"))
                compose!(cell, txt)
            end
            clr = tocolor(color(GWPos(x,y)), colormin, colormax)
            compose!(cell, Compose.rectangle(), fill(clr), Compose.stroke("gray"))
        end
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.5mm), cells...)
    outline = compose(context(), linewidth(1mm), Compose.rectangle(), Compose.stroke("gray"))

    if haskey(step, :s)
        agent = cell_ctx(step[:s], mdp.size)
        if haskey(step, :a)
            act = compose(context(), Compose.text(0.5, 0.5, aarrow[step[:a]], hcenter, vcenter), Compose.stroke("black"), fill("black"))
            compose!(agent, act)
        end
        compose!(agent, Compose.circle(0.5, 0.5, 0.4), fill(RGBA(1.0, 0.65, 0.0, 0.9) ))
    else
        agent = nothing
    end

    if haskey(step, :sp) && !isterminal(mdp, step[:sp])
        next_agent = cell_ctx(step[:sp], mdp.size)
        compose!(next_agent, Compose.circle(0.5, 0.5, 0.4), fill("lightblue"))
    else
        next_agent = nothing
    end

    sz = min(w, h)
    return compose(context((w - sz) / 2, (h - sz) / 2, sz, sz), agent, next_agent, grid, outline)
end

function cell_ctx(xy, size)
    nx, ny = size
    x, y = xy
    return context((x - 1) / nx, (ny - y) / ny, 1 / nx, 1 / ny)
end

tocolor(x, colormin, colormax) = x
function tocolor(r::Float64, colormin::Float64, colormax::Float64)
    frac = (r - colormin) / (colormax - colormin)
    return get(ColorSchemes.redgreensplit, frac)
end

tofunc(mdp::GeoSteeringMDP, f) = f
tofunc(mdp::GeoSteeringMDP, mat::AbstractMatrix) = s -> mat[s...]
tofunc(mdp::GeoSteeringMDP, v::AbstractVector) = s -> v[stateindex(mdp, s)]

const aarrow = Dict(:up => '↑', :left => '←', :down => '↓', :right => '→')

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
