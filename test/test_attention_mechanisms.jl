@testset "Attention Mechanisms" begin
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
        
        # Test mask values - upper triangular should be true
        for i in 1:seq_len
            for j in 1:seq_len
                @test mask[i, j] == (i <= j)
            end
        end
    end
end 