module GeoSteeringPOMDPs

using POMDPs
using POMDPTools
using POMDPModels
using Parameters
using StaticArrays
using Distributions
using Random
using Compose
using ColorSchemes

include("geosteering.jl")

export GeoSteering, 
       GWUniform,
       GWPos


include("visualization.jl")
export render     
       
end # module GeoSteeringPOMDPs

