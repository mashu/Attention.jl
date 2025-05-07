@testset "MultiHeadAttention" verbose=true begin
    # Test MultiHeadAttention construction
    @testset "Construction" begin
        # Test with default parameters
        mha = Attention.MultiHeadAttention(64)
        @test mha.nheads == 8
        @test mha.head_size == 8
        @test mha.q_proj.bias == false  # When bias=false, it should be false
        
        # Test with custom parameters
        mha = Attention.MultiHeadAttention(128, 16, bias=true, dropout_prob=0.2)
        @test mha.nheads == 16
        @test mha.head_size == 8
        @test mha.q_proj.bias isa AbstractArray  # When bias=true, it should be an array
        @test mha.attn_drop.p ≈ 0.2
    end
    
    # Test forward pass and output dimensions
    @testset "Forward pass" begin
        # Create test data
        d_model = 4
        seq_len = 3
        batch_size = 2
        nheads = 2
        
        # Create sample input
        X = rand(Float32, d_model, seq_len, batch_size)
        
        # Create MultiHeadAttention layer
        mha = Attention.MultiHeadAttention(d_model, nheads)
        
        # Test self-attention (1-argument version)
        output1, attention1 = mha(X)
        @test size(output1) == (d_model, seq_len, batch_size)
        @test size(attention1) == (seq_len, seq_len, nheads, batch_size)
        
        # Test key-value shared (2-argument version)
        output2, attention2 = mha(X, X)
        @test size(output2) == (d_model, seq_len, batch_size)
        @test size(attention2) == (seq_len, seq_len, nheads, batch_size)
        
        # Test full attention (3-argument version)
        output3, attention3 = mha(X, X, X)
        @test size(output3) == (d_model, seq_len, batch_size)
        @test size(attention3) == (seq_len, seq_len, nheads, batch_size)
        
        # All outputs should be identical since inputs are identical
        @test output1 ≈ output2
        @test output2 ≈ output3
    end
    
    # Test different attention implementations
    @testset "Attention implementations" begin
        d_model = 4
        seq_len = 3
        batch_size = 2
        
        # Create sample input
        X = rand(Float32, d_model, seq_len, batch_size)
        
        # Create MultiHeadAttention with different attention implementations
        mha1 = Attention.MultiHeadAttention(d_model, 2, attention_impl=Attention.DotProductAttention())
        mha2 = Attention.MultiHeadAttention(d_model, 2, attention_impl=Attention.NNlibAttention())
        
        # Both should run without errors
        output1, _ = mha1(X)
        output2, _ = mha2(X)
        
        # Both should have the same shape
        @test size(output1) == size(output2)
    end
    
    # Test with causal mask
    @testset "Causal masking" begin
        d_model = 4
        seq_len = 5
        batch_size = 2
        
        # Create sample input
        X = rand(Float32, d_model, seq_len, batch_size)
        
        # Create causal mask
        mask = Attention.make_causal_mask(X)
        
        # Create MultiHeadAttention layer
        mha = Attention.MultiHeadAttention(d_model, 2)
        
        # Apply attention with mask
        output, attention = mha(X, mask=mask)
        
        # Shape should be unchanged
        @test size(output) == (d_model, seq_len, batch_size)
        @test size(attention) == (seq_len, seq_len, 2, batch_size)
    end
end

@testset "Numerical equivalence with Flux" begin
    # Test equivalence with Flux.MultiHeadAttention
    # Define test parameters
    d_model = 64
    nheads = 8
    seq_len = 10
    batch_size = 4
    dropout_prob = 0.1 # Match Flux's dropout_p naming if different internally
    bias = true

    # Create sample input
    # Flux expects (embed_dim, seq_len, batch_size)
    # Our MHA also expects (d_model, seq_len, batch_size)
    x = rand(Float32, d_model, seq_len, batch_size)

    # Instantiate our MultiHeadAttention
    # We use NNlibAttention as Flux.MHA uses NNlib.dot_product_attention
    mha_custom = Attention.MultiHeadAttention(d_model, nheads,
                                            bias=bias,
                                            dropout_prob=dropout_prob,
                                            attention_impl=Attention.NNlibAttention())

    # Instantiate Flux.MultiHeadAttention
    # Flux.MultiHeadAttention implicitly uses NNlib.dot_product_attention
    # It expects embed_dim as a positional argument, and nheads, bias, dropout_prob as keyword arguments
    mha_flux = Flux.MultiHeadAttention(d_model; # d_model is positional
                                        nheads=nheads, # nheads is a keyword argument
                                        bias=bias,
                                        dropout_prob=dropout_prob) # dropout_prob is the correct keyword

    # Ensure same parameters are loaded into both models for a fair comparison
    # This means copying weights and biases from one to the other.
    # q_proj, k_proj, v_proj, out_proj weights and biases

    # Copy q_proj weights and biases
    mha_flux.q_proj.weight .= mha_custom.q_proj.weight
    if bias
        mha_flux.q_proj.bias .= mha_custom.q_proj.bias
    end

    # Copy k_proj weights and biases
    mha_flux.k_proj.weight .= mha_custom.k_proj.weight
    if bias
        mha_flux.k_proj.bias .= mha_custom.k_proj.bias
    end

    # Copy v_proj weights and biases
    mha_flux.v_proj.weight .= mha_custom.v_proj.weight
    if bias
        mha_flux.v_proj.bias .= mha_custom.v_proj.bias
    end

    # Copy out_proj weights and biases
    mha_flux.out_proj.weight .= mha_custom.out_proj.weight
    if bias
        mha_flux.out_proj.bias .= mha_custom.out_proj.bias
    end

    # Set both models to test mode to disable dropout
    Flux.testmode!(mha_custom)
    Flux.testmode!(mha_flux)

    @testset "Forward pass" begin
        # Our MHA returns (output, attention_weights)
        # Flux MHA also returns (output, attention_weights)
        output_custom, _ = mha_custom(x)
        output_flux, _ = mha_flux(x)

        @test size(output_custom) == size(output_flux)
        @test output_custom ≈ output_flux rtol=1e-5
    end

    @testset "Gradients" begin
        # Define a dummy loss function (sum of outputs)
        loss_custom(m, inp) = sum(first(m(inp)))
        loss_flux(m, inp) = sum(first(m(inp)))

        # Calculate gradients for our MHA
        gs_custom = Flux.gradient(loss_custom, mha_custom, x)
        
        # Calculate gradients for Flux MHA
        gs_flux = Flux.gradient(loss_flux, mha_flux, x)

        # Compare gradients for parameters
        # q_proj weights and biases
        @test gs_custom[1].q_proj.weight ≈ gs_flux[1].q_proj.weight rtol=1e-5
        if bias
            @test gs_custom[1].q_proj.bias ≈ gs_flux[1].q_proj.bias rtol=1e-5
        end

        # k_proj weights and biases
        @test gs_custom[1].k_proj.weight ≈ gs_flux[1].k_proj.weight rtol=1e-5
        if bias
            @test gs_custom[1].k_proj.bias ≈ gs_flux[1].k_proj.bias rtol=1e-5
        end

        # v_proj weights and biases
        @test gs_custom[1].v_proj.weight ≈ gs_flux[1].v_proj.weight rtol=1e-5
        if bias
            @test gs_custom[1].v_proj.bias ≈ gs_flux[1].v_proj.bias rtol=1e-5
        end

        # out_proj weights and biases
        @test gs_custom[1].out_proj.weight ≈ gs_flux[1].out_proj.weight rtol=1e-5
        if bias
            @test gs_custom[1].out_proj.bias ≈ gs_flux[1].out_proj.bias rtol=1e-5
        end

        # Compare gradients for input x
        @test gs_custom[2] ≈ gs_flux[2] rtol=1e-5
    end
end 