using POMDPs
using POMDPModels
using POMDPTools
using Random

include("geosteering.jl")

r = Dict(
        GWPos(4,3)=>-10.0, 
        GWPos(4,1)=>-5.0, 
        GWPos(3,3)=>10.0, 
        GWPos(2,4)=>3.0
    )
gw = SimpleGridWorld(
    size=(5, 5),
    rewards=r,
    terminate_from=Set(keys(r)),
    tprob=0.7,
    discount=0.95    
)

render(gw, (s=[2,2],))

gw.terminate_from