"""
    trackable.jl — Layer selection predicate via dispatch.

Determines which layers get wrapped with NaNCheck during nantrack.
Extend for custom leaf layers by adding a method.
"""

"""
    trackable(::KeyPath, layer) :: Bool

Predicate that decides whether `layer` should be wrapped with [`NaNCheck`](@ref).
Returns `true` for common Flux leaf layers (`Dense`, `Embedding`, `LayerNorm`,
`Scale`, `Conv`).

**Functions are not wrapped.** Pure functions (`relu`, `swish`, `identity`, etc.)
have no parameters and cannot introduce NaN through weights. Wrapping them
breaks GPU broadcasting (the `NaNCheck` wrapper is not `isbits`, which CUDA
kernels require) and interferes with libraries like Onion that store activation
functions as struct fields and broadcast them over GPU arrays.

Extend for your own leaf layers:

```julia
NaNTracker.trackable(::KeyPath, ::MyCustomLeaf) = true
```
"""
trackable(::KeyPath, ::Dense)           = true
trackable(::KeyPath, ::Flux.Embedding)  = true
trackable(::KeyPath, ::Flux.LayerNorm)  = true
trackable(::KeyPath, ::Flux.Scale)      = true
trackable(::KeyPath, ::Flux.Conv)       = true
trackable(::KeyPath, x)                 = false
