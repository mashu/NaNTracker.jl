# NaNTracker example: wrap a model and use it. No fmap_with_path or custom exclude.
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

# 3. Strip wrappers to get the original back.
restored = nanuntrack(tracked)
