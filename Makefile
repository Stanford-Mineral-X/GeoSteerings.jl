main:
	julia --project=. examples/main.jl

sim:
	julia --project=. examples/example.jl

all: main sim