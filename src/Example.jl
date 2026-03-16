# NaNTracker example: wrap a model, use stats tracking.
include("NaNTracker.jl")
using .NaNTracker
using Flux

# 1. Flux model: wrap and run.
model = Chain(Dense(4 => 8, relu), Dense(8 => 2))
tracked = nantrack(model)
x = randn(Float32, 4, 5)
y = tracked(x)   # same API as model(x)

# 2. Optional: for models with non-Flux leaf layers (e.g. Onion), extend trackable
#    (KeyPath is re-exported by NaNTracker), then nantrack(model):
#    using Onion
#    NaNTracker.trackable(::KeyPath, ::Onion.Linear) = true
#    NaNTracker.trackable(::KeyPath, ::Onion.RMSNorm) = true

# 3. Stats tracking: enable before a training step to diagnose explosions.
enable_stats!()
loss, grads = Flux.withgradient(tracked) do m
    sum(m(x))
end
dump_stats()                          # show all recent entries
dump_stats(path_contains="layers")    # filter by path substring
clear_stats!()                        # reset for next step
disable_stats!()                      # turn off when done debugging

# 4. Strip wrappers to get the original back.
restored = nanuntrack(tracked)
