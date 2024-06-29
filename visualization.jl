using Compose
using ColorSchemes

function render(mdp::GeoSteering, step::Union{NamedTuple,Dict}=(;);
    color=s -> reward(mdp, s),
    policy::Union{Policy,Nothing}=nothing,
    colormin::Float64=-10.0, colormax::Float64=10.0
)

    color = tofunc(mdp, color)

    nx, ny = mdp.size
    cells = []
    for x in 1:nx, y in 1:ny
        cell = cell_ctx((x, y), mdp.size)
        if GWPos(x, y) in keys(mdp.blocks)  # Check if the cell is a block
            compose!(cell, rectangle(), fill("black"), stroke("gray"))  # Render as a grey block
        else
            if policy !== nothing && !(GWPos(x, y) in mdp.terminate_from)  # Check if the cell is not a goal
                a = action(policy, GWPos(x,y))
                txt = compose(context(), text(0.5, 0.5, aarrow[a], hcenter, vcenter), stroke("black"), fill("black"))
                compose!(cell, txt)
            end
            clr = tocolor(color(GWPos(x,y)), colormin, colormax)
            compose!(cell, rectangle(), fill(clr), stroke("gray"))
        end
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.5mm), cells...)
    outline = compose(context(), linewidth(1mm), rectangle(), stroke("gray"))

    if haskey(step, :s)
        agent = cell_ctx(step[:s], mdp.size)
        if haskey(step, :a)
            act = compose(context(), text(0.5, 0.5, aarrow[step[:a]], hcenter, vcenter), stroke("black"), fill("black"))
            compose!(agent, act)
        end
        compose!(agent, circle(0.5, 0.5, 0.4), fill("orange"))
    else
        agent = nothing
    end

    if haskey(step, :sp) && !isterminal(mdp, step[:sp])
        next_agent = cell_ctx(step[:sp], mdp.size)
        compose!(next_agent, circle(0.5, 0.5, 0.4), fill("lightblue"))
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

tofunc(m::GeoSteering, f) = f
tofunc(m::GeoSteering, mat::AbstractMatrix) = s -> mat[s...]
tofunc(m::GeoSteering, v::AbstractVector) = s -> v[stateindex(m, s)]

const aarrow = Dict(:up => '↑', :left => '←', :down => '↓', :right => '→')