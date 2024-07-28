JULIA_PROJECT = --project=.
JULIA_BIN = $(shell which julia 2>/dev/null)
JULIA_VERSION = 1.9.3
JULIA_URL = https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-$(JULIA_VERSION)-linux-x86_64.tar.gz

.PHONY: install_julia install test format lint all

main:
	julia --project=. examples/main.jl

sim:
	julia --project=. examples/example.jl

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

install: instantiate
	julia $(JULIA_PROJECT) -e 'using Pkg; Pkg.instantiate()'

test:
	julia $(JULIA_PROJECT) test/test_geosteering.jl
	julia $(JULIA_PROJECT) test/test_mdp.jl
	
all: install test
