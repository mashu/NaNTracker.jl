module NaNTracker
    using Flux
    using Functors: KeyPath
    using ChainRulesCore

    __check_nan(x::AbstractArray) = any(isnan, x)
    __check_nan(x::Number) = isnan(x)
    __check_nan(x::Tuple) = any(__check_nan, x)
    __check_nan(x) = false

    struct DebugWrapper{L}
        path::KeyPath
        layer::L
    end

    function (debug::DebugWrapper)(args...; kwargs...)
        any(__check_nan.(args)) && throw(DomainError(args, "NaN on input for layer: $(debug.path)"))
        y = debug.layer(args...; kwargs...)
        __check_nan(y) && throw(DomainError(y, "NaN on output for layer: $(debug.path)"))
        return y
    end
    Flux.@layer DebugWrapper
    function ChainRulesCore.rrule(cfg::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode}, debug::DebugWrapper, x)
        y, pb = ChainRulesCore.rrule_via_ad(cfg, debug.layer, x)
        function pb_check(Δ)
            __check_nan(Δ) && throw(DomainError(Δ, "NaN on gradient input for layer: $(debug.path)"))
            gs = pb(Δ)
            __check_nan(gs) && throw(DomainError(Δ, "NaN on gradient output for layer: $(debug.path)"))
            return gs
        end
        return y, pb_check
    end

    function with_logging(f, args...)
        try
            f(args...)
        catch e
            open("error_log.txt", "a") do file
                println(file, "Error occurred: ", e)
            end
            throw(e)
        end
    end
    export with_logging
    export DebugWrapper
end # module NaNTracker
