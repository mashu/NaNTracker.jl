"""
    api.jl — Public API: nantrack / nanuntrack.

Walk the model tree and wrap/unwrap trackable leaf layers with NaNCheck.
"""

"""
    nantrack(model)

Wrap trackable leaf layers of `model` with [`NaNCheck`](@ref) for forward
and backward NaN detection.  Returns a structurally identical model that
throws `DomainError` (including the layer's `KeyPath`) at the first NaN.

The function uses `Functors.fmap_with_path` to walk the model tree and
only wraps layers for which [`trackable`](@ref) returns `true`.
Already-wrapped `NaNCheck` nodes are left unchanged (safe to call twice).

## Stats tracking

Enable `enable_stats!()` before a training step to record norm and maxabs
of every activation and gradient at each checked layer. When NaN is detected
the recent trajectory is printed automatically. Query stats at any time with
`dump_stats()` or `recent_stats()`.

See also [`nanuntrack`](@ref), [`trackable`](@ref), [`enable_stats!`](@ref).
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
