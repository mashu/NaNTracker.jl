# Usage

```julia
include("NaNTracker.jl")
using .NaNTracker
using Flux
using Flux.Functors: KeyPath

#
# First, we define a simple encoder only model
#
struct EncoderOnly
    embedding::Embedding
    mha::MultiHeadAttention
    mha_norm::LayerNorm
end
Flux.@layer EncoderOnly
function EncoderOnly(vocab_size::Int, hidden_size::Int, head_size::Int, nheads::Int, dropout::Float64)
    embedding = Embedding(vocab_size => hidden_size)
    mha = MultiHeadAttention(hidden_size => head_size => hidden_size, nheads=nheads, dropout_prob=dropout)
    mha_norm = LayerNorm(hidden_size)
    return EncoderOnly(embedding, mha, mha_norm)
end
function (g::EncoderOnly)(x; attn_mask=nothing)
    z̄ = g.embedding(x)
    z̄ = g.mha_norm(first(g.mha(z̄, mask=attn_mask)) + z̄)
    return z̄
end

enc = EncoderOnly(30, 128, 64, 2, 0.1)


#
# Second, we wrap suspected layers (or all layers) model with NaNTracker
#

wrapped_embedding = DebugWrapper(KeyPath("Embedding"), enc.embedding)
wrapped_mha = DebugWrapper(KeyPath("MultiHeadAttention"), enc.mha)
wrapped_mha_norm = DebugWrapper(KeyPath("LayerNorm"), enc.mha_norm)

# Define a wrapper for the MWE model that uses the wrapped layers
struct WrappedMWE
    embedding::DebugWrapper{typeof(enc.embedding)}
    mha::DebugWrapper{typeof(enc.mha)}
    mha_norm::DebugWrapper{typeof(enc.mha_norm)}
end
Flux.@layer WrappedMWE
function (wmwe::WrappedMWE)(x; attn_mask=nothing)
    z̄ = wmwe.embedding(x)
    z̄, _ = wmwe.mha(z̄, z̄, z̄; mask=attn_mask) # Adjust according to MultiHeadAttention API
    z̄ = wmwe.mha_norm(z̄) + z̄
    return z̄
end
# Create a wrapped instance of your model
wrapped_model = WrappedMWE(wrapped_embedding, wrapped_mha, wrapped_mha_norm)

# Test the model
x = map(f->rand(Int32.(2:10), rand(8:16)), 1:32)
x = reduce(hcat, rpad.(x, maximum(length.(x)), 1)) #|> gpu
# Input array broadcastable to size (kv_len, q_len, nheads, batch_size)
mask = permutedims(repeat((x .== 1), outer = [1, 1, 1, 1]), (1, 4, 3, 2))

loss, grads = Flux.withgradient(wrapped_model) do m
    sum(m(x, attn_mask=mask))
end
```
