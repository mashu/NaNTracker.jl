"""
    nancheck.jl — NaN-checking wrapper for Flux layers.

NaNCheck is a plain struct registered with Functors.@functor so that
fmap / Optimisers can reach the parameters inside `layer`.

Property forwarding: accessing a field that is not `:path` or `:layer`
delegates to the wrapped layer. This makes NaNCheck transparent to
libraries that inspect layer internals (e.g. Onion.StarGLU reads
`dense.weight` to extract raw weight matrices).  Zygote correctly
traces through `getfield(nc, :layer)` → `getproperty(layer, s)`,
so gradient flow is preserved.
"""

"""
    NaNCheck{P,L}

Thin wrapper around a Flux layer that checks for NaN on every forward and
backward pass.  `P` is the path type (`KeyPath`), `L` is the wrapped
layer type.

This struct:
- Has **no custom `rrule`** — the inner layer is differentiated normally
  by whatever AD backend is active.
- Forwards `getproperty` for unknown fields to the wrapped layer, making
  it transparent to code that accesses layer internals (e.g. `.weight`).
- Is registered with `Functors.@functor` (not `Flux.@layer`) so that
  `fmap` / `Optimisers.update!` can reach the trainable parameters
  inside `layer`.
"""
struct NaNCheck{P,L}
    path::P
    layer::L
end

Functors.@functor NaNCheck

function Base.getproperty(nc::NaNCheck, s::Symbol)
    s === :path && return getfield(nc, :path)
    s === :layer && return getfield(nc, :layer)
    return getproperty(getfield(nc, :layer), s)
end

function (n::NaNCheck)(args...; kwargs...)
    checked = map(a -> nanbarrier(getfield(n, :path), "forward input", a), args)
    y = getfield(n, :layer)(checked...; kwargs...)
    return nanbarrier(getfield(n, :path), "forward output", y)
end
