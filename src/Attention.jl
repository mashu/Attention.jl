module Attention

using Flux, NNlib, LinearAlgebra

# Export abstract types and interfaces
export AbstractAttention
export compute_attention

# Export concrete attention mechanisms
export DotProductAttention, NNlibAttention

# Export MultiHeadAttention
export MultiHeadAttention

# Export utility functions
export make_causal_mask

# Include the attention mechanism implementations
include("AttentionMechanisms.jl")

# Include the MultiHeadAttention implementation
include("MultiHead.jl")

end # module 