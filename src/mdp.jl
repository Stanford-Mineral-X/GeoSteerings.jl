function POMDPs.states(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP})
    cells = [[Cell(x, y) for x in 1:mdp.size[1], y in 1:mdp.size[2]]...]
    push!(cells, Cell(-1, -1))
    
    bool_vals = [false for _ in 1:mdp.size[1]]   
    valid_surf_stats = [deepcopy(bool_vals) for _ in 1:mdp.size[1]+1]
    push!(deepcopy(valid_surf_stats), bool_vals)
    
    for i in 1:mdp.size[1]
        valid_surf_stats[i+1][1:i] .= true
    end

    surrounding_stats = [get_surrounding_status(mdp, cell) for cell in cells[1:end-1]]    

    ss = Vector{State}(undef, length(valid_surf_stats) * length(surrounding_stats) * length(cells))
    index = 1
    for cell in cells
        for surf_stat in valid_surf_stats
            for surrounding_stat in surrounding_stats
                ss[index] = State(cell, surf_stat, surrounding_stat)
                index += 1
            end            
        end
    end

    return ss
end


function POMDPs.stateindex(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, s::State)
    id = findfirst(==(s), states(mdp))
    if isnothing(id)
        error("State $s not found in states.")        
        return nothing
    end
    return id
end



function POMDPs.isterminal(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, s::State)    
    return s.cell in mdp.terminal_zone || s.cell[1] == mdp.size[1]
end


function POMDPs.initialstate(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}) 
    y_min_target, y_max_target = get_target_bounds(mdp, 1)
    is_surface_visited = [false for _ in 1:mdp.size[1]]
    is_surface_visited[1] = true
    initial_states = [State(Cell(1, y), is_surface_visited, get_surrounding_status(mdp, Cell(1,y))) for y in y_min_target:y_max_target]
    return SparseCat(initial_states, fill(1/length(initial_states), length(initial_states)))
end


POMDPs.actions(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}) = (UP, DOWN, RIGHT)
POMDPs.actionindex(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, a::Action) = aind[a]

function POMDPs.transition(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, s::State, a::Action)
    num_actions = length(actions(mdp))
    destinations = Array{State}(undef, num_actions)

    probs = Array(zeros(num_actions))

    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = 1 - mdp.drift_prob # probability of transitioning to the desired State
        else
            prob = mdp.drift_prob / (num_actions - 1) # probability of transitioning to another State
        end

        dest = move(mdp, s, act)
        probs[i] += prob
        

        if !inbounds(mdp, dest.cell) # hit an edge and come back            
            destinations[i] = s # dest was out of bounds
        else
            destinations[i] = dest
        end
    end        

    return SparseCat(destinations, probs)

end

function POMDPs.reward(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, s::State, a::Action, sp::State)
    if isterminal(mdp, s)
        return 0.0
    end

    if sp.cell in mdp.target_zone 
        if !s.is_surface_visited[sp.cell[1]] # reward for entering the target zone surface first time
            return mdp.reward_target  
        end
    end
    return mdp.reward_offtarget
end

function POMDPs.reward(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, s::State, a::Action)
    if isterminal(mdp, s)
        return 0.0
    end

    sp = rand(transition(mdp, s, a))

    if sp.cell in mdp.target_zone 
        if !s.is_surface_visited[sp.cell[1]] # reward for entering the target zone surface first time
            return mdp.reward_target  
        end
    end
    return mdp.reward_offtarget
end

function POMDPs.discount(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP})
   return mdp.discount
end
