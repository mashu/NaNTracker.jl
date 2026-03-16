# NaNTracker.jl

[![CI](https://github.com/mashu/NaNTracker.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mashu/NaNTracker.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/mashu/NaNTracker.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mashu/NaNTracker.jl)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://mashu.github.io/NaNTracker.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://mashu.github.io/NaNTracker.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Lightweight NaN detection for [Flux.jl](https://github.com/FluxML/Flux.jl) models.
Wraps leaf layers to check forward inputs, forward outputs, gradient inputs, and gradient outputs — throws a `DomainError` with the exact layer path at the first NaN.

## Quick start

```julia
using NaNTracker, Flux

model = Chain(Dense(10 => 20, relu), Dense(20 => 5))

# Wrap — checks every forward and backward pass for NaN
tracked = nantrack(model)

loss, grads = Flux.withgradient(tracked) do m
    sum(m(x))
end

# Unwrap when done debugging
clean = nanuntrack(tracked)
```
