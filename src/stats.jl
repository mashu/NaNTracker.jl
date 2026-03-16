"""
    stats.jl — Activation/gradient stats collection for NaN diagnosis.

Opt-in system: disabled by default with zero overhead (dispatch to no-op).
When enabled, records norm, maxabs, NaN/Inf flags at every NaNCheck layer
into a fixed-capacity ring buffer. On NaN detection the trajectory is
dumped automatically to show where values exploded.

Uses `AbstractStatsCollector` hierarchy instead of Union types:
- `DisabledCollector` : no-op (default, zero overhead)
- `ActiveCollector`   : ring buffer that records `StatsEntry` per observation
"""

# ─── Tensor statistics (dispatch-based, GPU-safe) ────────────────────────────
#
# On GPU arrays the reductions launch kernels and the Float64 conversion
# triggers a device→host sync — acceptable overhead for debugging.

"""Frobenius norm of a numeric array."""
tensor_norm(x::AbstractArray{<:Number}) = isempty(x) ? 0.0 : Float64(sqrt(sum(abs2, x)))
tensor_norm(x::Number) = Float64(abs(x))
tensor_norm(x::Tuple) = sqrt(sum(tensor_norm(a)^2 for a in x))
tensor_norm(x) = 0.0

"""Maximum absolute value in a numeric array."""
tensor_maxabs(x::AbstractArray{<:Number}) = isempty(x) ? 0.0 : Float64(maximum(abs, x))
tensor_maxabs(x::Number) = Float64(abs(x))
tensor_maxabs(x::Tuple) = isempty(x) ? 0.0 : maximum(tensor_maxabs(a) for a in x)
tensor_maxabs(x) = 0.0

"""Whether any element is ±Inf."""
tensor_has_inf(x::AbstractArray{<:Number}) = any(isinf, x)
tensor_has_inf(x::Number) = isinf(x)
tensor_has_inf(x::Tuple) = any(tensor_has_inf, x)
tensor_has_inf(x) = false

# ─── Stats entry ─────────────────────────────────────────────────────────────

"""One recorded observation of a tensor flowing through a NaNCheck layer."""
struct StatsEntry
    path::String
    label::String
    norm::Float64
    maxabs::Float64
    has_nan::Bool
    has_inf::Bool
end

# ─── Collector hierarchy (dispatch replaces Union{Nothing, ...}) ─────────────

"""Abstract base for stats collectors. Dispatch selects no-op vs active recording."""
abstract type AbstractStatsCollector end

"""Disabled collector: all operations are no-ops. Default state."""
struct DisabledCollector <: AbstractStatsCollector end

"""
    ActiveCollector(capacity)

Fixed-capacity ring buffer of `StatsEntry`. Overwrites oldest entries when full.
"""
mutable struct ActiveCollector <: AbstractStatsCollector
    entries::Vector{StatsEntry}
    capacity::Int
    pos::Int      # next write position (1-based, wraps)
    count::Int    # total entries written (capped at capacity for retrieval)
end

ActiveCollector(capacity::Int) = ActiveCollector(Vector{StatsEntry}(undef, capacity), capacity, 1, 0)

# ─── Ring buffer operations ──────────────────────────────────────────────────

"""Record one stats entry into the ring buffer."""
function push_entry!(col::ActiveCollector, entry::StatsEntry)
    @inbounds col.entries[col.pos] = entry
    col.pos = mod1(col.pos + 1, col.capacity)
    col.count = min(col.count + 1, col.capacity)
    nothing
end

"""Return the last `n` entries in chronological order (oldest first)."""
function recent_entries(col::ActiveCollector, n::Int)
    n = min(n, col.count)
    n == 0 && return StatsEntry[]
    result = Vector{StatsEntry}(undef, n)
    for i in 1:n
        idx = mod1(col.pos - n + i - 1, col.capacity)
        @inbounds result[i] = col.entries[idx]
    end
    result
end

# ─── Recording (dispatch: disabled = no-op, active = record) ─────────────────

"""No-op: stats collection disabled."""
record_stats!(::DisabledCollector, path, label, x) = nothing

"""Record norm, maxabs, NaN/Inf flags for tensor `x` at the given path and label."""
function record_stats!(col::ActiveCollector, path, label, x)
    push_entry!(col, StatsEntry(
        string(path), label,
        tensor_norm(x), tensor_maxabs(x),
        hasnan(x), tensor_has_inf(x),
    ))
end

# ─── Formatting ──────────────────────────────────────────────────────────────

function format_stat_value(v::Float64)
    isnan(v)   && return "NaN"
    isinf(v)   && return v > 0 ? "+Inf" : "-Inf"
    abs(v) < 1e-3 ? @sprintf("%.2e", v) :
    abs(v) > 1e6  ? @sprintf("%.2e", v) :
                     @sprintf("%.4f", v)
end

function format_entry(io::IO, e::StatsEntry; index::Int = 0)
    flags = ""
    e.has_nan && (flags *= " ⚠ NaN")
    e.has_inf && (flags *= " ⚠ Inf")
    idx_str = index > 0 ? lpad(index, 4) * "  " : ""
    print(io, idx_str)
    print(io, rpad(e.path, 42))
    print(io, rpad(e.label, 22))
    print(io, "norm=", lpad(format_stat_value(e.norm), 12))
    print(io, "  maxabs=", lpad(format_stat_value(e.maxabs), 12))
    println(io, flags)
end

function format_entries_table(io::IO, entries::Vector{StatsEntry})
    println(io, "      ", rpad("path", 42), rpad("label", 22), lpad("norm", 16), lpad("maxabs", 21))
    for (i, e) in enumerate(entries)
        format_entry(io, e; index = i)
    end
end

# ─── NaN context for error messages (dispatch: disabled = no-op) ─────────────

"""No-op: nothing to emit when stats are disabled."""
emit_nan_context(::DisabledCollector) = nothing

"""Dump recent stats trajectory to stderr when NaN is detected."""
function emit_nan_context(col::ActiveCollector)
    col.count == 0 && return nothing
    entries = recent_entries(col, min(40, col.count))
    io = IOBuffer()
    println(io)
    println(io, "══ Recent activation/gradient stats (oldest → newest) ══")
    format_entries_table(io, entries)
    println(io, "══ end stats ══")
    @error String(take!(io))
    nothing
end

# ─── Global collector ────────────────────────────────────────────────────────

const STATS_COLLECTOR = Ref{AbstractStatsCollector}(DisabledCollector())

# ─── Public API ──────────────────────────────────────────────────────────────

"""
    enable_stats!(; capacity=1000)

Turn on activation/gradient stats collection. Each forward input, forward output,
and gradient flowing through a `NaNCheck` layer records norm, maxabs, and
NaN/Inf flags into a ring buffer.

**Note:** On GPU this introduces sync points (scalar transfers) at every checked
layer. Use for debugging only.
"""
enable_stats!(; capacity::Int = 1000) = (STATS_COLLECTOR[] = ActiveCollector(capacity); nothing)

"""Turn off stats collection and release the buffer."""
disable_stats!() = (STATS_COLLECTOR[] = DisabledCollector(); nothing)

"""Clear recorded stats without disabling collection."""
clear_stats!(::DisabledCollector) = nothing
clear_stats!(col::ActiveCollector) = (STATS_COLLECTOR[] = ActiveCollector(col.capacity); nothing)
clear_stats!() = clear_stats!(STATS_COLLECTOR[])

"""
    recent_stats(; n=50, path_contains="")

Return recent `StatsEntry` records. Optionally filter by path substring.
Returns empty vector when stats are disabled.
"""
recent_stats(::DisabledCollector; n::Int = 50, path_contains::AbstractString = "") = StatsEntry[]

function recent_stats(col::ActiveCollector; n::Int = 50, path_contains::AbstractString = "")
    entries = recent_entries(col, min(n * 4, col.count))  # fetch extra before filtering
    if !isempty(path_contains)
        entries = filter(e -> occursin(path_contains, e.path), entries)
    end
    last(entries, min(n, length(entries)))
end

recent_stats(; n::Int = 50, path_contains::AbstractString = "") =
    recent_stats(STATS_COLLECTOR[]; n, path_contains)

"""
    dump_stats(; n=50, path_contains="", io=stderr)

Print recent stats entries to `io`. Useful for inspecting activation/gradient
magnitudes during training without waiting for a NaN.

# Example
```julia
enable_stats!()
# ... run one training step ...
dump_stats(path_contains="attention")  # show only attention layers
clear_stats!()                          # reset for next step
```
"""
function dump_stats(::DisabledCollector; n::Int = 50, path_contains::AbstractString = "", io::IO = stderr)
    println(io, "No stats recorded (stats disabled — call enable_stats!()).")
    nothing
end

function dump_stats(col::ActiveCollector; n::Int = 50, path_contains::AbstractString = "", io::IO = stderr)
    entries = recent_stats(col; n, path_contains)
    if isempty(entries)
        println(io, "No stats recorded.")
        return nothing
    end
    println(io, "─── Activation / gradient stats (oldest → newest) ───")
    format_entries_table(io, entries)
    println(io, "─── $(length(entries)) entries shown ───")
    nothing
end

dump_stats(; n::Int = 50, path_contains::AbstractString = "", io::IO = stderr) =
    dump_stats(STATS_COLLECTOR[]; n, path_contains, io)
