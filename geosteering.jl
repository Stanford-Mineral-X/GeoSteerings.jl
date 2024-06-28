using StaticArrays

const GWPos = SVector{2,Int}


#Create a new struct called GeoSteering that is a GeoSteering with non-navigation cells at the boundaries
#inherit from SimpleGridWorld

Base.@kwdef struct GeoSteering <: MDP{GWPos, Symbol}
    size::Tuple{Int, Int}           = (10,10)
    rewards::Dict{GWPos, Float64}   = Dict(GWPos(4,3)=>-10.0, GWPos(4,6)=>-5.0, GWPos(9,3)=>10.0, GWPos(8,8)=>3.0)
    faults::Set{GWPos}               = Set([GWPos(1,1), GWPos(1,2), GWPos(1,3), GWPos(1,4), GWPos(1,5), GWPos(1,6), GWPos(1,7), GWPos(1,8), GWPos(1,9), GWPos(1,10), GWPos(2,1), GWPos(2,10), GWPos(3,1), GWPos(3,10), GWPos(4,1), GWPos(4,10), GWPos(5,1), GWPos(5,10), GWPos(6,1), GWPos(6,10), GWPos(7,1), GWPos(7,10), GWPos(8,1), GWPos(8,10), GWPos(9,1), GWPos(9,10), GWPos(10,1), GWPos(10,2), GWPos(10,3), GWPos(10,4), GWPos(10,5), GWPos(10,6), GWPos(10,7), GWPos(10,8), GWPos(10,9), GWPos(10,10)])
    terminate_from::Set{GWPos}      = Set(keys(rewards))
    tprob::Float64                  = 0.7
    discount::Float64               = 0.95
end

# States
function POMDPs.states(mdp::GeoSteering)
    ss = vec(GWPos[GWPos(x, y) for x in 1:mdp.size[1], y in 1:mdp.size[2]])
    push!(ss, GWPos(-1,-1))
    return ss
end

function POMDPs.stateindex(mdp::GeoSteering, s::AbstractVector{Int})
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

POMDPs.initialstate(mdp::GeoSteering) = GWUniform(mdp.size)


# Actions

POMDPs.actions(mdp::GeoSteering) = (:up, :down, :left, :right)
Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L = t[rand(rng, 1:length(t))] # don't know why this doesn't work out of the box


const dir = Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0))
const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4)

POMDPs.actionindex(mdp::GeoSteering, a::Symbol) = aind[a]


# Transitions

POMDPs.isterminal(m::GeoSteering, s::AbstractVector{Int}) = any(s.<0)

function POMDPs.transition(mdp::GeoSteering, s::AbstractVector{Int}, a::Symbol)
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

function inbounds(m::GeoSteering, s::AbstractVector{Int})
    #considers faults as boundaries
    return 1 <= s[1] <= m.size[1] && 1 <= s[2] <= m.size[2] && !(GWPos(s[1], s[2]) in m.faults)
end





# Rewards

POMDPs.reward(mdp::GeoSteering, s::AbstractVector{Int}) = get(mdp.rewards, s, 0.0)
POMDPs.reward(mdp::GeoSteering, s::AbstractVector{Int}, a::Symbol) = reward(mdp, s)


# discount

POMDPs.discount(mdp::GeoSteering) = mdp.discount

# Conversion
function POMDPs.convert_a(::Type{V}, a::Symbol, m::GeoSteering) where {V<:AbstractArray}
    convert(V, [aind[a]])
end
function POMDPs.convert_a(::Type{Symbol}, vec::V, m::GeoSteering) where {V<:AbstractArray}
    actions(m)[convert(Int, first(vec))]
end