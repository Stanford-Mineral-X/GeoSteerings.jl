function POMDPs.states(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP})
    cells = [[Cell(x, y) for x in 1:mdp.size[1], y in 1:mdp.size[2]]...]
    push!(cells, Cell(-1, -1))
    

    surrounding_stats = [get_surrounding_status(mdp, cell) for cell in cells[1:end-1]]    

    ss = Vector{State}(undef,  length(surrounding_stats) * length(cells))
    index = 1
    for cell in cells        
        for surrounding_stat in surrounding_stats
            ss[index] = State(cell, surrounding_stat)
            index += 1
        end
    end

    return ss
end


function POMDPs.stateindex(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, s::State)
    id = findfirst(==(s), POMDPs.states(mdp))
    if isnothing(id)
        println("State $s not found in states.")        
        return nothing
    end
    return id
end



function POMDPs.isterminal(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, s::State)    
    return s.cell in mdp.terminal_zone
end


function POMDPs.initialstate(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}) 
    y_min_target, y_max_target = get_target_bounds(mdp, 1)
    initial_states = [State(Cell(1, y), get_surrounding_status(mdp, Cell(1,y))) for y in y_min_target:y_max_target]
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
        if sp.cell in mdp.terminal_zone
            return mdp.reward_goal        
        end
        return mdp.reward_target
    end
    return mdp.reward_offtarget
end


function POMDPs.gen(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP}, s::State, a::Action, rng::AbstractRNG)
    sp = rand(rng, transition(mdp, s, a))
    r = reward(mdp, s, a, sp)
    return (sp=sp, r=r)
end


function POMDPs.discount(mdp::Union{GeoSteeringMDP, GeoSteeringPOMDP})
   return mdp.discount
end
