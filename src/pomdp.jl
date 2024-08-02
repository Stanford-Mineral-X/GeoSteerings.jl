function POMDPs.initialize_belief(pomdp::GeoSteeringPOMDP)
    return initialstate(pomdp)
end

function POMDPs.observations(pomdp::GeoSteeringPOMDP)
    cells = [[Cell(x, y) for x in 1:pomdp.size[1], y in 1:pomdp.size[2]]...]
    push!(cells, Cell(-1, -1))

    surrounding_stats = [get_surrounding_status_combinations(pomdp, cell) for cell in cells[1:end-1]]
    surrounding_stats = [el_ for stat_cell in surrounding_stats for el_ in stat_cell]
    
    observations = Vector{Observation}(undef, length(cells)*length(surrounding_stats))
    index = 1
    for cell in cells
        for surrounding in surrounding_stats
            observations[index] = Observation(surrounding)
            index += 1
        end
    end
    return observations
end

Base.:(==)(o1::Observation, o2::Observation) = o1.is_surrounding_target == o2.is_surrounding_target 

function POMDPs.obsindex(pomdp::GeoSteeringPOMDP, o::Observation)
    id = findfirst(==(o), observations(pomdp))
    if isnothing(id)
        error("Observation $o not found in observations.")        
        return nothing
    end
    return id
end

function POMDPs.observation(pomdp::GeoSteeringPOMDP, sp::State)
    obs_true = Observation(sp.is_surrounding_target)
    surrounding_stats = get_surrounding_status_combinations(pomdp, sp.cell)
    obs_list = Vector{Observation}(undef, length(surrounding_stats))
    base_prob = 1/length(surrounding_stats)
    obs_probs = fill(base_prob, length(surrounding_stats))

    index = 1
    for surrounding_stat in surrounding_stats
        obs = Observation(surrounding_stat)
        obs_list[index] = deepcopy(obs)
        
        
        if very_similar(obs, obs_true)
            obs_probs[index] = pomdp.similar_obs_prob_multiplier*base_prob
        end
        index += 1
    end

    obs_probs = obs_probs ./ sum(obs_probs)

    return SparseCat(obs_list, obs_probs)
end

function POMDPs.gen(pomdp::GeoSteeringPOMDP, s::State, a::Action, rng::AbstractRNG)
    sp = rand(rng, transition(pomdp, s, a))
    obs = rand(rng, observation(pomdp, s, a, sp))
    r = reward(pomdp, s, a, sp)
    return (sp=sp, r=r, o=obs)
end


