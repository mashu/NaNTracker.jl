# Guide

## How it works

`nantrack` uses [`Functors.fmap_with_path`](https://fluxml.ai/Functors.jl/stable/)
to walk the model tree. Each leaf layer that satisfies [`trackable`](@ref) gets
wrapped in a [`NaNCheck`](@ref) node.

The wrapper does two things:

1. **Forward pass** — before calling the wrapped layer it checks
   `any(hasnan, args)`. After the call it checks `hasnan(y)`. If either
   fires, a `DomainError` is thrown with the layer's `KeyPath`.

2. **Backward pass** — a custom `ChainRulesCore.rrule` intercepts
   automatic differentiation. It checks the incoming gradient `Δ` and
   the pullback output for NaN, throwing with the same path information.

Because `NaNCheck` is parametric (`NaNCheck{P,L}`), it is fully
type-stable and works transparently on CPU and GPU arrays.

## Composite models

Composite structs (anything registered with `Flux.@layer`) are **not**
wrapped themselves — `fmap` recurses into them so only the leaf
computational layers get checked. This means you define your model
exactly as before:

```julia
struct Encoder
    embedding::Embedding
    mha::MultiHeadAttention
    norm::LayerNorm
end
Flux.@layer Encoder

function (e::Encoder)(x; mask=nothing)
    z = e.embedding(x)
    z = e.norm(first(e.mha(z, mask=mask)) + z)
    return z
end

model = Encoder(Embedding(100 => 64), MultiHeadAttention(64 => 32 => 64, nheads=4), LayerNorm(64))
tracked = nantrack(model)  # Dense, Embedding, LayerNorm inside get wrapped
```

## Custom leaf layers

By default the following are tracked: `Dense`, `Embedding`, `LayerNorm`,
`Scale`, `Conv`, and `Function` (activations).

To add your own leaf layer, define a single dispatch method:

```julia
struct MyAttention
    W::Dense
end
Flux.@layer MyAttention
(m::MyAttention)(x) = softmax(m.W(x))

# Option A: track MyAttention as a whole
NaNTracker.trackable(::KeyPath, ::MyAttention) = true

# Option B: leave it as false (default) so fmap recurses into W,
# and Dense is already tracked.
```

## Unwrapping

Call [`nanuntrack`](@ref) to strip all `NaNCheck` nodes and restore the
original model. This is useful when you are done debugging and want to
deploy without overhead:

```julia
clean = nanuntrack(tracked)
```

## GPU support

`NaNCheck` does not constrain element types or array types, so it works
transparently with CUDA arrays:

```julia
using CUDA
gpu_model = nantrack(model) |> gpu
```
