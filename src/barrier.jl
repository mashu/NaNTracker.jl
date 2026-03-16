"""
    barrier.jl — NaN barrier: identity that checks forward & gradient.

Instead of writing a custom rrule for the *wrapped layer* (which required
rrule_via_ad, tangent assembly, and knowledge of the AD backend), we
insert thin identity functions whose only job is to inspect values as
they flow past.

• Forward:  optionally record stats, check x, return x unchanged.
• Backward: optionally record stats, check Δ, return Δ unchanged.

The real layer keeps its own AD rules — whatever backend is in use
(Zygote, Enzyme, Mooncake, …) differentiates it without interference.
"""

"""Emit stats context (if enabled) then throw DomainError."""
function throw_nan_error(path, message)
    emit_nan_context(STATS_COLLECTOR[])
    throw(DomainError(path, message))
end

"""
    nanbarrier(path, label, x)

Identity function with a ChainRules rrule that checks for NaN in the
forward value `x` and in the incoming gradient `Δ`.  Throws `DomainError`
with the layer `path` and a human-readable `label` on detection.

When stats are enabled (`enable_stats!()`), records norm and maxabs of `x`
and `Δ` into a ring buffer. On NaN detection the recent stats trajectory
is dumped to stderr before throwing, showing the explosion path.

Works transparently on CPU and GPU arrays (uses `any(isnan, x)` which
CUDA.jl lowers to a device reduction).
"""
function nanbarrier(path, label, x)
    record_stats!(STATS_COLLECTOR[], path, label, x)
    hasnan(x) && throw_nan_error(path, "NaN in $label")
    return x
end

function ChainRulesCore.rrule(::typeof(nanbarrier), path, label, x)
    record_stats!(STATS_COLLECTOR[], path, label, x)
    hasnan(x) && throw_nan_error(path, "NaN in $label")
    function nanbarrier_pullback(Δ)
        record_stats!(STATS_COLLECTOR[], path, "∇ $label", Δ)
        hasnan(Δ) && throw_nan_error(path, "NaN in gradient at $label")
        return NoTangent(), NoTangent(), NoTangent(), Δ
    end
    return x, nanbarrier_pullback
end
