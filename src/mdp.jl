const GWPos = SVector{2,Int}


#Create a new struct called GeoSteering that is a GeoSteering with non-navigation cells at the boundaries
#inherit from SimpleGridWorld

@with_kw mutable struct GeoSteeringMDP <: MDP{GWPos, Symbol}
    size::Tuple{Int, Int}           = (10,10)
    goals::Dict{GWPos, Float64}   = Dict(
        GWPos(10,6)=> 10, GWPos(10,5)=>10, #8 up up, 9 down right
    )
    rocks::Dict{GWPos, Float64} = Dict(
        GWPos(4,4)=>-3.0, GWPos(5, 4)=>-3.0, 
        GWPos(6, 4)=>-3.0, GWPos(7, 4)=>-3.0, 
        GWPos(8, 4)=>-3.0, GWPos(9, 4)=>-3.0,
        )
    hard_rocks::Dict{GWPos, Float64} = Dict(
        GWPos(5,5)=>-6.0, GWPos(6, 5)=>-6.0, GWPos(7, 5)=>-6.0, GWPos(8, 5)=>-6.0, 
        GWPos(7, 6)=>-6.0, GWPos(8, 6)=>-6.0,
        GWPos(8, 7)=>-6.0,
        )
    blocks::Vector{GWPos} = [
        GWPos(1, 1), GWPos(1, 2), GWPos(1,3), GWPos(1,4), GWPos(1,7), GWPos(1,8), GWPos(1,9), GWPos(1,10),
        GWPos(2, 1), GWPos(2, 2), GWPos(2,3), GWPos(2,4), GWPos(2,7), GWPos(2,8), GWPos(2,9), GWPos(2,10),
        GWPos(3, 1), GWPos(3, 2), GWPos(3,3), GWPos(3,4), GWPos(3,7), GWPos(3,8), GWPos(3,9), GWPos(3,10),
        GWPos(4, 1), GWPos(4, 2), GWPos(4,3), GWPos(4,8), GWPos(4,9), GWPos(4,10),
        GWPos(5, 1), GWPos(5, 2), GWPos(5,3), GWPos(5,8), GWPos(5,9), GWPos(5,10),
        GWPos(6, 1), GWPos(6, 2), GWPos(6,3), GWPos(6,9), GWPos(6,10),
        GWPos(7, 1), GWPos(7, 2), GWPos(7,3), GWPos(7,9), GWPos(7,10),
        GWPos(8, 1), GWPos(8, 2), GWPos(8,3), GWPos(8,9), GWPos(8,10),
        GWPos(9, 1), GWPos(9, 2), GWPos(9,3), GWPos(9,9), GWPos(9,10),
        GWPos(10,1), GWPos(10,2), GWPos(10,3), GWPos(10,4), GWPos(10,5), GWPos(10,7), GWPos(10,8), GWPos(10,9), GWPos(10,10),
    ]
    rewards::Dict{GWPos, Float64}   = merge(goals, rocks, hard_rocks)
    terminate_from::Set{GWPos}      = Set(keys(goals))
    tprob::Float64                  = 1.0
    discount::Float64               = 0.95 
end

GeoSteering() = GeoSteeringMDP()

# States
function POMDPs.states(mdp::GeoSteeringMDP)
    ss = vec(GWPos[GWPos(x, y) for x in 1:mdp.size[1], y in 1:mdp.size[2]])
    push!(ss, GWPos(-1,-1))
    return ss
end

function POMDPs.stateindex(mdp::GeoSteeringMDP, s::AbstractVector{Int})
    if all(s.>0)
        return LinearIndices(mdp.size)[s...]
    else
        return prod(mdp.size) + 1 # TODO: Change
    end
end


struct GWUniform
    size::Tuple{Int, Int}
end


Base.rand(rng::AbstractRNG, d::GWUniform) = GWPos(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]))
function POMDPs.pdf(d::GWUniform, s::GWPos)
    if all(1 .<= s[1] .<= d.size)
        return 1/prod(d.size)
    else
        return 0.0
    end
end
POMDPs.support(d::GWUniform) = (GWPos(x, y) for x in 1:d.size[1], y in 1:d.size[2])

# POMDPs.initialstate(mdp::GeoSteeringMDP) = GWUniform(mdp.size)
# initial state can only be at GWPos(1,5) or GWPos(2,5)
POMDPs.initialstate(mdp::GeoSteeringMDP) = Deterministic(GWPos(1,5))


# Actions

POMDPs.actions(mdp::GeoSteeringMDP) = (:up, :down, :left, :right)
Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L = t[rand(rng, 1:length(t))] # don't know why this doesn't work out of the box


const dir = Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0))
const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4)

POMDPs.actionindex(mdp::GeoSteeringMDP, a::Symbol) = aind[a]


# Transitions

POMDPs.isterminal(mdp::GeoSteeringMDP, s::AbstractVector{Int}) = any(s.<0)

function POMDPs.transition(mdp::GeoSteeringMDP, s::AbstractVector{Int}, a::Symbol)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return Deterministic(GWPos(-1,-1))
    end

    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s

    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = mdp.tprob # probability of transitioning to the desired cell
        else
            prob = (1.0 - mdp.tprob)/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + dir[act]
        destinations[i+1] = dest

        if !inbounds(mdp, dest) # hit an edge and come back
            probs[1] += prob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(convert(SVector, destinations), convert(SVector, probs))
end

function inbounds(mdp::GeoSteeringMDP, s::AbstractVector{Int})
    #considers blocks as boundaries    
    return 1 <= s[1] <= mdp.size[1] && 1 <= s[2] <= mdp.size[2] && !([s[1], s[2]] in mdp.blocks)

end




# Rewards

POMDPs.reward(mdp::GeoSteeringMDP, s::AbstractVector{Int}) = get(mdp.rewards, s, -1.0)
POMDPs.reward(mdp::GeoSteeringMDP, s::AbstractVector{Int}, a::Symbol) = reward(mdp, s)


# discount

POMDPs.discount(mdp::GeoSteeringMDP) = mdp.discount

# Conversion
function POMDPs.convert_a(::Type{V}, a::Symbol, mdp::GeoSteeringMDP) where {V<:AbstractArray}
    convert(V, [aind[a]])
end
function POMDPs.convert_a(::Type{Symbol}, vec::V, mdp::GeoSteeringMDP) where {V<:AbstractArray}
    actions(mdp)[convert(Int, first(vec))]
end
