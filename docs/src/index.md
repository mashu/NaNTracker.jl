# NaNTracker.jl

Lightweight NaN detection for [Flux.jl](https://github.com/FluxML/Flux.jl) models.

NaNTracker wraps leaf layers to check forward inputs, forward outputs,
and incoming gradients — throwing a `DomainError` with the exact
layer path at the first NaN.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/mashu/NaNTracker.jl")
```

## Quick start

```julia
using NaNTracker, Flux

model = Chain(Dense(10 => 20, relu), Dense(20 => 5))

# Wrap — every forward and backward pass is checked for NaN
tracked = nantrack(model)

x = randn(Float32, 10, 8)
loss, grads = Flux.withgradient(tracked) do m
    sum(m(x))
end

# Remove tracking when done
clean = nanuntrack(tracked)
```

If a NaN appears anywhere in the computation, you get:

```
DomainError with KeyPath(:layers, 2):
NaN in forward output
```
