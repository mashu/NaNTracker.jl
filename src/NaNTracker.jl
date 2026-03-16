module NaNTracker

using Flux
using Functors
using ChainRulesCore: NoTangent
import ChainRulesCore

export nantrack, nanuntrack, trackable, NaNCheck, KeyPath

# ─── NaN detection via dispatch ───────────────────────────────────────────────

"""
    hasnan(x) :: Bool

Check whether `x` contains any NaN values. Dispatches on type so it works
for arrays, scalars, tuples, and falls back to `false` for anything else.
"""
hasnan(x::AbstractArray) = any(isnan, x)
hasnan(x::Number)        = isnan(x)
hasnan(x::Tuple)         = any(hasnan, x)
hasnan(x)                = false

# ─── NaN barrier: identity that checks forward & gradient ─────────────────────
#
# Instead of writing a custom rrule for the *wrapped layer* (which required
# rrule_via_ad, tangent assembly, and knowledge of the AD backend), we
# insert thin identity functions whose only job is to inspect values as
# they flow past.
#
# • Forward:  check x, return x unchanged.
# • Backward: check Δ, return Δ unchanged.
#
# The real layer keeps its own AD rules — whatever backend is in use
# (Zygote, Enzyme, Mooncake, …) differentiates it without interference.

"""
    nanbarrier(path, label, x)

Identity function with a ChainRules rrule that checks for NaN in the
forward value `x` and in the incoming gradient `Δ`.  Throws `DomainError`
with the layer `path` and a human-readable `label` on detection.

Works transparently on CPU and GPU arrays (uses `any(isnan, x)` which
CUDA.jl lowers to a device reduction).
"""
function nanbarrier(path, label, x)
    hasnan(x) && throw(DomainError(path, "NaN in $label"))
    return x
end

function ChainRulesCore.rrule(::typeof(nanbarrier), path, label, x)
    hasnan(x) && throw(DomainError(path, "NaN in $label"))
    function nanbarrier_pullback(Δ)
        hasnan(Δ) && throw(DomainError(path, "NaN in gradient at $label"))
        return NoTangent(), NoTangent(), NoTangent(), Δ
    end
    return x, nanbarrier_pullback
end

# ─── NaN-checking wrapper ────────────────────────────────────────────────────
#
# NaNCheck is a plain struct registered with Functors.@functor so that
# fmap / Optimisers can reach the parameters inside `layer`.
#
# Property forwarding: accessing a field that is not `:path` or `:layer`
# delegates to the wrapped layer. This makes NaNCheck transparent to
# libraries that inspect layer internals (e.g. Onion.StarGLU reads
# `dense.weight` to extract raw weight matrices).  Zygote correctly
# traces through `getfield(nc, :layer)` → `getproperty(layer, s)`,
# so gradient flow is preserved.

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

# ─── Layer selection via dispatch ────────────────────────────────────────────

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

# ─── Public API ──────────────────────────────────────────────────────────────

"""
    nantrack(model)

Wrap trackable leaf layers of `model` with [`NaNCheck`](@ref) for forward
and backward NaN detection.  Returns a structurally identical model that
throws `DomainError` (including the layer's `KeyPath`) at the first NaN.

The function uses `Functors.fmap_with_path` to walk the model tree and
only wraps layers for which [`trackable`](@ref) returns `true`.
Already-wrapped `NaNCheck` nodes are left unchanged (safe to call twice).

See also [`nanuntrack`](@ref), [`trackable`](@ref).
"""
function nantrack(model)
    should_stop(kp, x) = trackable(kp, x) || x isa NaNCheck
    fmap_with_path(model; exclude = should_stop) do kp, x
        trackable(kp, x) ? NaNCheck(kp, x) : x
    end
end

"""
    nanuntrack(model)

Strip all [`NaNCheck`](@ref) wrappers, restoring the original model.
"""
function nanuntrack(model)
    fmap(model; exclude = x -> x isa NaNCheck) do x
        x isa NaNCheck ? getfield(x, :layer) : x
    end
end

end # module NaNTracker
