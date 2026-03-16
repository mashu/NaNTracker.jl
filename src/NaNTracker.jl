module NaNTracker

using Flux
using Functors: KeyPath, fmap_with_path, fmap
using ChainRulesCore: RuleConfig, HasReverseMode, rrule_via_ad, NoTangent, Tangent
import ChainRulesCore

export nantrack, nanuntrack, trackable, NaNCheck, KeyPath

# ─── NaN detection via dispatch ───────────────────────────────────────────────

"""
    hasnan(x) :: Bool

Check whether `x` contains any NaN values. Dispatches on type so it works
for arrays, scalars, tuples, and falls back to `false` for anything else.
"""
hasnan(x::AbstractArray) = any(isnan, x)
hasnan(x::Number) = isnan(x)
hasnan(x::Tuple) = any(hasnan, x)
hasnan(x) = false

# ─── NaN-checking wrapper ────────────────────────────────────────────────────

"""
    NaNCheck{P,L}

Thin wrapper around a Flux layer that checks for NaN on every forward call.
`P` is the path type (`KeyPath`), `L` is the wrapped layer type.
Parametric so it stays type-stable on both CPU and GPU.
"""
struct NaNCheck{P,L}
    path::P
    layer::L
end

Flux.@layer NaNCheck

function (n::NaNCheck)(args...; kwargs...)
    any(hasnan, args) && throw(DomainError(n.path, "NaN in forward input"))
    y = n.layer(args...; kwargs...)
    hasnan(y) && throw(DomainError(n.path, "NaN in forward output"))
    return y
end

# ─── Backward pass NaN checking ──────────────────────────────────────────────

function ChainRulesCore.rrule(cfg::RuleConfig{>:HasReverseMode}, n::NaNCheck, args...)
    any(hasnan, args) && throw(DomainError(n.path, "NaN in forward input"))
    y, pb = rrule_via_ad(cfg, n.layer, args...)
    hasnan(y) && throw(DomainError(n.path, "NaN in forward output"))

    function nancheck_pullback(Δ)
        hasnan(Δ) && throw(DomainError(n.path, "NaN in gradient input"))
        gs = pb(Δ)
        any(hasnan, gs) && throw(DomainError(n.path, "NaN in gradient output"))
        ∂layer = first(gs)
        ∂args  = Base.tail(gs)
        return (Tangent{typeof(n)}(; path = NoTangent(), layer = ∂layer), ∂args...)
    end

    return y, nancheck_pullback
end

# ─── Layer selection via dispatch ────────────────────────────────────────────

"""
    trackable(::KeyPath, layer) :: Bool

Predicate that decides whether `layer` should be wrapped with [`NaNCheck`](@ref).
Returns `true` for Flux leaf layers (`Dense`, `Embedding`, `LayerNorm`,
`Scale`, `Conv`, `Function`) and `false` otherwise so that `fmap` recurses
into composite layers.

Extend for your own leaf layers:

```julia
NaNTracker.trackable(::KeyPath, ::MyCustomLeaf) = true
```
"""
trackable(::KeyPath, ::Dense) = true
trackable(::KeyPath, ::Flux.Embedding) = true
trackable(::KeyPath, ::Flux.LayerNorm) = true
trackable(::KeyPath, ::Flux.Scale) = true
trackable(::KeyPath, ::Flux.Conv) = true
trackable(::KeyPath, ::Function) = true
trackable(::KeyPath, x) = false

# Values that appear inside layers (e.g. Dropout.dims::Colon) are passed through
# unchanged; we never wrap them. Uses dispatch only (no type checks).
pass_through(::Colon) = true
pass_through(::Number) = true
pass_through(::Symbol) = true
pass_through(::Bool) = true
pass_through(::Nothing) = true
pass_through(x) = false

# ─── Public API ──────────────────────────────────────────────────────────────

"""
    nantrack(model; exclude = trackable)

Wrap leaf layers of `model` with [`NaNCheck`](@ref) for forward and backward
NaN detection. Returns a structurally identical model that throws
`DomainError` (including the layer's `KeyPath`) at the first NaN.

Usage: `tracked = nantrack(model)` then use `tracked` like `model`. For models
that use non-Flux leaf layers (e.g. Onion.Linear, Onion.RMSNorm), extend
[`trackable`](@ref) with one line per leaf type.

See also [`nanuntrack`](@ref), [`trackable`](@ref).
"""
function nantrack(model; exclude = trackable)
    wrap_if_trackable(kp, x) = pass_through(x) ? x : (trackable(kp, x) ? NaNCheck(kp, x) : x)
    stop_at_trackable_or_pass(kp, x) = exclude(kp, x) || pass_through(x)
    return fmap_with_path(wrap_if_trackable, model; exclude = stop_at_trackable_or_pass)
end

"""
    nanuntrack(model)

Strip all [`NaNCheck`](@ref) wrappers, restoring the original model.
"""
nanuntrack(model) = fmap(unwrap, model; exclude = is_nancheck)

unwrap(n::NaNCheck) = n.layer
unwrap(x) = x

is_nancheck(::NaNCheck) = true
is_nancheck(x) = false

end # module NaNTracker
