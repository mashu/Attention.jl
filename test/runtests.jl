using Test
using Attention
using Flux
using LinearAlgebra

@testset "Attention.jl" verbose=true begin
    # Run the attention mechanics tests
    include("test_attention_mechanisms.jl")
    
    # Run the MultiHeadAttention tests
    include("test_multihead.jl")
end 