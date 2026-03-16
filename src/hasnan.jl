"""
    hasnan.jl — NaN detection via dispatch.

Type-based NaN detection that works for arrays, scalars, tuples, and
falls back to `false` for anything else. GPU-safe: `any(isnan, x)`
lowers to a device reduction on CUDA arrays.
"""

"""
    hasnan(x) :: Bool

Check whether `x` contains any NaN values. Dispatches on type so it works
for arrays, scalars, tuples, and falls back to `false` for anything else.
"""
hasnan(x::AbstractArray) = any(isnan, x)
hasnan(x::Number)        = isnan(x)
hasnan(x::Tuple)         = any(hasnan, x)
hasnan(x)                = false
