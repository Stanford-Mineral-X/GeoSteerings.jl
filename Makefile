JULIA_PROJECT = --project=.
JULIA_BIN = $(shell which julia 2>/dev/null)
JULIA_VERSION = 1.9.3
JULIA_URL = https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-$(JULIA_VERSION)-linux-x86_64.tar.gz


main:
	julia --project=. examples/main.jl

mdp:
	julia --project=. examples/mdp_example.jl

pomdp:
	julia --project=. examples/pomdp_example.jl


install_julia:
ifeq ($(JULIA_BIN),)
	@echo "Julia not found. Installing Julia..."
	curl -L $(JULIA_URL) -o julia.tar.gz
	tar -xzf julia.tar.gz
	sudo mv julia-$(JULIA_VERSION) /opt/julia
	sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia
	rm julia.tar.gz
else
	@echo "Julia is already installed at $(JULIA_BIN)"
endif

install:
	julia $(JULIA_PROJECT) -e 'using Pkg; Pkg.instantiate()'

test:
	julia $(JULIA_PROJECT) --code-coverage test/runtests.jl
	
all: install test
