module GeoSteerings

using POMDPs
using POMDPTools
using POMDPModels
using Parameters
using StaticArrays
using Distributions
using Random
using Compose
using ColorSchemes
using Colors
using POMDPs
using POMDPTools
using POMDPModels
using Parameters
using Cairo
using Fontconfig



export 
       GeoSteering, 
       GWUniform,
       GWPos
include("mdp.jl")       

export 
       render, 
       savefig
include("visualization.jl")       
       
end # module GeoSteerings

