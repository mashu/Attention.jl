@testset "Attention Mechanisms" verbose=true begin
    # Test dimensions and values for DotProductAttention
    @testset "DotProductAttention" begin
        # Create input tensors
        d_model = 4
        seq_len_q = 3
        seq_len_k = 3
        batch_size = 2
        
        # Create sample inputs
        q = rand(Float32, d_model, seq_len_q, batch_size)
        k = rand(Float32, d_model, seq_len_k, batch_size)
        v = rand(Float32, d_model, seq_len_k, batch_size)
        
        # Compute attention
        mechanism = Attention.DotProductAttention()
        output, attention = Attention.compute_attention(mechanism, q, k, v)
        
        # Test output shape
        @test size(output) == (d_model, seq_len_q, batch_size)
        @test size(attention) == (seq_len_k, seq_len_q, 1, batch_size)
        
        # Test with multiple heads
        nheads = 2
        output_multi, attention_multi = Attention.compute_attention(mechanism, q, k, v; nheads=nheads)
        @test size(output_multi) == (d_model, seq_len_q, batch_size)
        @test size(attention_multi) == (seq_len_k, seq_len_q, nheads, batch_size)
    end
    
    # Test numerical equivalence between DotProductAttention and NNlib.dot_product_attention
    @testset "Numerical Equivalence" begin
        # Test with different configurations
        for (d_model, seq_len, batch_size, nheads) in [
            (4, 3, 2, 1),    # Single head
            (8, 4, 2, 2),    # Multiple heads
            (16, 5, 3, 4),   # Larger model
        ]
            # Create sample inputs
            q = rand(Float32, d_model, seq_len, batch_size)
            k = rand(Float32, d_model, seq_len, batch_size)
            v = rand(Float32, d_model, seq_len, batch_size)
            
            # Compute attention using both implementations
            mechanism = Attention.DotProductAttention()
            output_custom, attention_custom = Attention.compute_attention(mechanism, q, k, v; nheads=nheads)
            output_nnlib, attention_nnlib = NNlib.dot_product_attention(q, k, v; nheads=nheads)
            
            # Test numerical equivalence
            @test isapprox(output_custom, output_nnlib, rtol=1e-5)
            @test isapprox(attention_custom, attention_nnlib, rtol=1e-5)
            
            # Test with causal mask
            bool_mask = Attention.make_causal_mask(q)
            
            output_custom_masked, attention_custom_masked = Attention.compute_attention(mechanism, q, k, v; mask=bool_mask, nheads=nheads)
            output_nnlib_masked, attention_nnlib_masked = NNlib.dot_product_attention(q, k, v; mask=bool_mask, nheads=nheads)
            
            # Test numerical equivalence with mask
            @test isapprox(output_custom_masked, output_nnlib_masked, rtol=1e-5)
            @test isapprox(attention_custom_masked, attention_nnlib_masked, rtol=1e-5)
        end
    end
    
    # Test NNlibAttention (which currently falls back to DotProductAttention)
    @testset "NNlibAttention" begin
        d_model = 4
        seq_len = 3
        batch_size = 2
        
        # Create sample inputs
        q = rand(Float32, d_model, seq_len, batch_size)
        k = rand(Float32, d_model, seq_len, batch_size)
        v = rand(Float32, d_model, seq_len, batch_size)
        
        # Compute attention
        mechanism = Attention.NNlibAttention()
        output, attention = Attention.compute_attention(mechanism, q, k, v)
        
        # Test output shape
        @test size(output) == (d_model, seq_len, batch_size)
        @test size(attention) == (seq_len, seq_len, 1, batch_size)
    end
    
    # Test causal mask creation
    @testset "make_causal_mask" begin
        # Create a dummy tensor
        x = rand(4, 5, 2)  # (d_model, seq_len, batch)
        
        # Create mask for dimension 2 (seq_len)
        mask = Attention.make_causal_mask(x)
        
        # Test mask shape
        seq_len = size(x, 2)
        @test size(mask) == (seq_len, seq_len)
        
        # Test mask values - upper triangle (including diagonal) should be true
        # and lower triangle should be false
        expected = UpperTriangular(ones(Bool, seq_len, seq_len))
        @test mask == expected
    end
end 