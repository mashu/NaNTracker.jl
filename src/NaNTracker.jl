"""
    NaNTracker

Lightweight NaN detection for Flux.jl models. Wraps leaf layers to check
forward inputs, forward outputs, and incoming gradients — throws a
`DomainError` with the exact layer path at the first NaN.

Optional stats tracking (`enable_stats!()`) records norm and maxabs at
every checked layer, showing where values explode before the NaN.
"""
module NaNTracker

using Flux
using Functors
using ChainRulesCore: NoTangent
import ChainRulesCore
using Printf: @sprintf

# ── Source files (order matters: each file may depend on earlier ones) ────────

include("hasnan.jl")      # hasnan dispatch
include("stats.jl")       # AbstractStatsCollector, recording, formatting, public stats API
include("barrier.jl")     # nanbarrier + rrule (depends on hasnan, stats)
include("nancheck.jl")    # NaNCheck struct (depends on barrier)
include("trackable.jl")   # trackable dispatch (uses Flux layer types)
include("api.jl")         # nantrack, nanuntrack (depends on NaNCheck, trackable)

# ── Exports ──────────────────────────────────────────────────────────────────

export nantrack, nanuntrack, trackable, NaNCheck, KeyPath
export enable_stats!, disable_stats!, clear_stats!, recent_stats, dump_stats

end # module NaNTracker
