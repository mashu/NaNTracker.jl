include("NaNTracker.jl")
using .NaNTracker
using Flux
using Functors
using Functors: KeyPath, fmap_with_path

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

#
# Second, we wrap model with DebugWrapper
#
exclude(kp::KeyPath, x::Dense) = true
exclude(kp::KeyPath, x::Function) = false
exclude(kp::KeyPath, x) = false

debug_model(model) = Functors.fmap_with_path(DebugWrapper, model, exclude = exclude)
enc = debug_model(EncoderOnly(30, 128, 64, 2, 0.1))

# Test the model
x = map(f->rand(Int32.(2:10), rand(8:16)), 1:32)
x = reduce(hcat, rpad.(x, maximum(length.(x)), 1))
# Input array broadcastable to size (kv_len, q_len, nheads, batch_size)
mask = permutedims(repeat((x .== 1), outer = [1, 1, 1, 1]), (1, 4, 3, 2))

loss, grads = Flux.withgradient(enc) do m
    sum(m(x, attn_mask=mask))
end

