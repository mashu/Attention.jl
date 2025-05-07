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