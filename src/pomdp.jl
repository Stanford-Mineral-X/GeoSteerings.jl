function POMDPs.initialize_belief(pomdp::GeoSteeringPOMDP)
    ss = states(pomdp)
    ss = [s for s in ss if s.cell[1] == 1]

    return SparseCat(ss, fill(1/length(ss), length(ss)))
end

function POMDPs.observations(pomdp::GeoSteeringPOMDP)
    cells = [[Cell(x, y) for x in 1:pomdp.size[1], y in 1:pomdp.size[2]]...]
    push!(cells, Cell(-1, -1))
    is_surrounding_target = [get_surrounding_status(pomdp, cell) for cell in cells[1:end-1]]
    
    observations = Vector{Observation}(undef, length(is_surrounding_target) * length(cells))
    index = 1
    for cell in cells
        for surrounding in is_surrounding_target
            observations[index] = Observation(cell, surrounding)
            index += 1
        end
    end
    return observations
end

Base.:(==)(o1::Observation, o2::Observation) = o1.cell == o2.cell && o1.is_surrounding_target == o2.is_surrounding_target

function POMDPs.obsindex(pomdp::GeoSteeringPOMDP, o::Observation)
    id = findfirst(==(o), observations(pomdp))
    if isnothing(id)
        error("Observation $o not found in observations.")        
        return nothing
    end
    return id
end

function POMDPs.observation(pomdp::GeoSteeringPOMDP, s::State, a::Action, sp::State)
    cells = [[Cell(x, y) for x in 1:pomdp.size[1], y in 1:pomdp.size[2]]...]
    measurements_list = [get_surrounding_status(pomdp, cell) for cell in cells]

    observations = Vector{Observation}(undef, length(measurements_list))
    index = 1
    for surrounding in measurements_list
        observations[index] = Observation(sp.cell, surrounding)
        index += 1
    end

    return SparseCat(observations, fill(1/length(observations), length(observations)))
end

function POMDPs.gen(pomdp::GeoSteeringPOMDP, s::State, a::Action, rng::AbstractRNG)
    sp = rand(rng, transition(pomdp, s, a))
    obs = rand(rng, observation(pomdp, s, a, sp))
    r = reward(pomdp, s, a, sp)
    return (sp=sp, r=r, o=obs)
end


