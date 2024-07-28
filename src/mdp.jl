function POMDPs.states(mdp::GeoSteeringMDP)
    cells = [[Cell(x, y) for x in 1:mdp.size[1], y in 1:mdp.size[2]]...]
    push!(cells, Cell(-1, -1))
    is_surface_visited = [b for b in Iterators.product(fill([false, true], mdp.size[1])...)]

    num_cells = length(cells)
    num_surfaces = length(is_surface_visited)
    ss = Vector{State}(undef, num_cells * num_surfaces)

    index = 1
    for cell in cells
        for surface in is_surface_visited
            ss[index] = State(cell, [surface...])
            index += 1
        end
    end

    return ss
end



function POMDPs.stateindex(mdp::GeoSteeringMDP, s::State)
    cell_index = if all(s.cell .> 0)
        LinearIndices(mdp.size)[s.cell.x, s.cell.y]
    else
        prod(mdp.size) + 1
    end

    surface_index = sum(Int.(s.is_surface_visited) .* (2 .^ (0:(length(s.is_surface_visited) - 1))))

    return (cell_index - 1) * 2^mdp.size[1] + surface_index + 1
end


function POMDPs.isterminal(mdp::GeoSteeringMDP, s::State)    
    return s.cell in mdp.terminal_zone
end


function POMDPs.initialstate(mdp::GeoSteeringMDP) 
    y_min_target, y_max_target = get_target_bounds(mdp, 1)
    is_surface_visited = [false for _ in 1:mdp.size[1]]
    is_surface_visited[1] = true
    initial_states = [State(Cell(1, y), is_surface_visited) for y in y_min_target:y_max_target]
    return SparseCat(initial_states, fill(1/length(initial_states), length(initial_states)))
end


POMDPs.actions(mdp::GeoSteeringMDP) = (UP, DOWN, RIGHT)
POMDPs.actionindex(mdp::GeoSteeringMDP, a::Action) = aind[a]

function POMDPs.transition(mdp::GeoSteeringMDP, s::State, a::Action)
    num_actions = length(actions(mdp))
    destinations = Array{State}(undef, num_actions)

    probs = Array(zeros(num_actions))

    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = 1 - mdp.drift_prob # probability of transitioning to the desired State
        else
            prob = mdp.drift_prob / (num_actions - 1) # probability of transitioning to another State
        end

        dest = s + act
        probs[i] += prob

        if !inbounds(mdp, dest.cell) # hit an edge and come back            
            destinations[i] = s # dest was out of bounds
        else
            destinations[i] = dest
        end
    end        

    return SparseCat(destinations, probs)

end

function POMDPs.reward(mdp::GeoSteeringMDP, s::State, a::Action, sp::State)
    if isterminal(mdp, s)
        return 0.0
    end

    if sp in mdp.target_zone 
        if !s.is_surface_visited[sp.cell[1]]
            return mdp.reward_target
        else
            return mdp.reward_redundant_target
        end
    end
    return mdp.reward_offtarget
end

function POMDPs.discount(mdp::GeoSteeringMDP)
   return mdp.discount
end
