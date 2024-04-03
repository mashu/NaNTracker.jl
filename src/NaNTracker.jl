module NaNTracker
    using Flux
    using Flux.Functors: fmap_with_path, KeyPath
    using Flux.ChainRulesCore

    function __check_nan(x::AbstractArray)
        return any(isnan, x)
    end

    function __check_nan(x::Number)
        return isnan(x)
    end

    function __check_nan(x::Tuple)
        return any(__check_nan, x)
    end

    function __check_nan(x)
        return false
    end

    struct DebugWrapper{L}
        path::KeyPath
        layer::L
    end
    function (debug::DebugWrapper)(x)
        __check_nan(x) && throw(DomainError(x, "NaN on input for layer: $(debug.path)"))
        y = debug.layer(x)
        __check_nan(y) && throw(DomainError(y, "NaN on output for layer: $(debug.path)"))
        return y
    end
    function (debug::DebugWrapper)(args...; kwargs...)
        any(__check_nan.(args)) && throw(DomainError(args, "NaN on input for layer: $(debug.path)"))
        y = debug.layer(args...; kwargs...)
        __check_nan(y) && throw(DomainError(y, "NaN on output for layer: $(debug.path)"))
        return y
    end
    Flux.@layer DebugWrapper
    function ChainRulesCore.rrule(cfg::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
        debug::DebugWrapper, x)
        y, pb = ChainRulesCore.rrule_via_ad(cfg, debug.layer, x)
        function pb_check(Δ)
            __check_nan(Δ) && throw(DomainError(Δ, "NaN on gradient input for layer: $(debug.path)"))
            gs = pb(Δ)
            __check_nan(gs) && throw(DomainError(Δ, "NaN on gradient output for layer: $(debug.path)"))
            return gs
        end
        return y, pb_check
    end

    export DebugWrapper
end # module NaNTracker
