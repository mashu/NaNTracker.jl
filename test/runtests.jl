using Test
using Random
using NaNTracker
using Flux
using Functors: KeyPath

@testset "NaNTracker.jl" begin

    @testset "hasnan" begin
        @test NaNTracker.hasnan(Float32[1.0, 2.0, NaN]) == true
        @test NaNTracker.hasnan(Float32[1.0, 2.0, 3.0]) == false
        @test NaNTracker.hasnan(NaN) == true
        @test NaNTracker.hasnan(1.0) == false
        @test NaNTracker.hasnan((Float32[1.0], Float32[NaN])) == true
        @test NaNTracker.hasnan((Float32[1.0], Float32[2.0])) == false
        @test NaNTracker.hasnan("not a number") == false
        @test NaNTracker.hasnan(nothing) == false
    end

    @testset "NaNCheck forward" begin
        layer = Dense(3 => 2)
        checked = NaNCheck(KeyPath(:test), layer)

        # Clean input passes through
        x = Float32[1.0, 2.0, 3.0]
        y = checked(x)
        @test size(y) == (2,)

        # NaN input throws
        bad = Float32[1.0, NaN, 3.0]
        @test_throws DomainError checked(bad)
    end

    @testset "NaNCheck forward output" begin
        # Layer that produces NaN output
        layer = Dense(2 => 2)
        layer.weight .= NaN
        checked = NaNCheck(KeyPath(:nan_weight), layer)
        @test_throws DomainError checked(Float32[1.0, 2.0])
    end

    @testset "NaNCheck kwargs forwarding" begin
        mha = MultiHeadAttention(8 => 4 => 8, nheads=2)
        checked = NaNCheck(KeyPath(:mha), mha)
        x = randn(Float32, 8, 3, 1)
        y = checked(x; mask=nothing)
        @test !isnothing(y)
    end

    @testset "nantrack wrapping" begin
        model = Chain(Dense(4 => 8, relu), Dense(8 => 2))
        tracked = nantrack(model)

        # Wrapped layers are NaNCheck
        @test tracked.layers[1] isa NaNCheck
        @test tracked.layers[2] isa NaNCheck

        # Forward pass works
        x = randn(Float32, 4, 5)
        y = tracked(x)
        @test size(y) == (2, 5)

        # Primitives inside layers (e.g. Dropout.dims::Colon) are not wrapped
        chain_with_dropout = Chain(Dense(2 => 2), Flux.Dropout(0.1))
        tracked_dropout = nantrack(chain_with_dropout)
        @test tracked_dropout.layers[1] isa NaNCheck
        @test tracked_dropout.layers[2] isa Flux.Dropout
        @test tracked_dropout.layers[2].dims === (:)  # Colon passed through, not wrapped
    end

    @testset "nanuntrack restores model" begin
        model = Chain(Dense(4 => 8), Dense(8 => 2))
        tracked = nantrack(model)
        restored = nanuntrack(tracked)

        @test restored.layers[1] isa Dense
        @test restored.layers[2] isa Dense

        # Same weights
        @test restored.layers[1].weight == model.layers[1].weight
        @test restored.layers[2].weight == model.layers[2].weight
    end

    @testset "gradient computation" begin
        model = Chain(Dense(4 => 8, relu), Dense(8 => 2))
        tracked = nantrack(model)
        x = randn(Float32, 4, 5)

        loss, grads = Flux.withgradient(tracked) do m
            sum(m(x))
        end

        @test isfinite(loss)
        @test !isnothing(grads)
    end

    @testset "gradient NaN detection" begin
        # A layer whose backward produces NaN is hard to construct
        # in a unit test, but we can at least verify the path works
        # by checking NaN input detection during gradient
        model = Dense(3 => 2)
        tracked = nantrack(model)
        bad = Float32[1.0, NaN, 3.0]

        @test_throws DomainError Flux.withgradient(tracked) do m
            sum(m(bad))
        end
    end

    @testset "trackable dispatch" begin
        kp = KeyPath()
        @test trackable(kp, Dense(3 => 2)) == true
        @test trackable(kp, Flux.Embedding(10 => 8)) == true
        @test trackable(kp, Flux.LayerNorm(8)) == true
        @test trackable(kp, relu) == true
        @test trackable(kp, Chain()) == false
        @test trackable(kp, "something") == false
    end

    @testset "custom trackable extension" begin
        struct MyLayer
            w::Matrix{Float32}
        end
        Flux.@layer MyLayer
        (m::MyLayer)(x) = m.w * x

        # Not tracked by default
        model = Chain(MyLayer(randn(Float32, 2, 3)))
        tracked = nantrack(model)
        @test !(tracked.layers[1] isa NaNCheck)

        # After extending trackable
        NaNTracker.trackable(::KeyPath, ::MyLayer) = true
        tracked2 = nantrack(model)
        @test tracked2.layers[1] isa NaNCheck

        x = randn(Float32, 3, 4)
        @test size(tracked2(x)) == (2, 4)
    end

    @testset "composite model" begin
        struct MLP{F1,F2}
            fc1::F1
            fc2::F2
        end
        Flux.@layer MLP
        (m::MLP)(x) = m.fc2(relu.(m.fc1(x)))

        model = MLP(Dense(4 => 8), Dense(8 => 2))
        tracked = nantrack(model)

        # Composite struct is not wrapped, leaf Dense layers are
        @test tracked isa MLP
        @test tracked.fc1 isa NaNCheck
        @test tracked.fc2 isa NaNCheck

        x = randn(Float32, 4, 5)
        y = tracked(x)
        @test size(y) == (2, 5)

        loss, grads = Flux.withgradient(tracked) do m
            sum(m(x))
        end
        @test isfinite(loss)
    end

    @testset "encoder model with MultiHeadAttention" begin
        struct Encoder{E,M,N}
            embedding::E
            mha::M
            norm::N
        end
        Flux.@layer Encoder

        function Encoder(vocab::Int, dim::Int, hdim::Int, nheads::Int)
            Encoder(Embedding(vocab => dim),
                    MultiHeadAttention(dim => hdim => dim, nheads=nheads),
                    LayerNorm(dim))
        end

        function (e::Encoder)(x; mask=nothing)
            z = e.embedding(x)
            z = e.norm(first(e.mha(z, mask=mask)) + z)
            return z
        end

        # Wrapping + correct mask shape; forward and backward with mask passed.
        Random.seed!(42)
        # Smaller model (1 head, small dims) so forward with mask is stable.
        model = Encoder(10, 8, 4, 1)
        tracked = nantrack(model)
        Flux.testmode!(tracked)

        # Padded batch: (seqlen, batch). Padding token = 1.
        seqs = [Int32[2, 3], Int32[3, 4], Int32[2], Int32[4, 5, 6]]
        maxlen = 3
        x = reduce(hcat, [vcat(s, fill(Int32(1), maxlen - length(s))) for s in seqs])
        # NNlib mask: broadcastable to (kv_len, q_len, nheads, batch). true = attend, false = mask out.
        key_padding = (x .== 1)  # true where padding
        mask = reshape(.!(key_padding), maxlen, 1, 1, size(x, 2))
        @test size(mask) == (maxlen, 1, 1, 4)

        y = tracked(x, mask=mask)
        @test size(y, 1) == 8
        @test size(y, 2) == maxlen
        loss, grads = Flux.withgradient(tracked) do m
            sum(m(x, mask=mask))
        end
        @test isfinite(loss)
    end

end
