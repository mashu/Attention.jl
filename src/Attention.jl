"""
    Attention

A Julia package providing modular and extensible attention mechanisms for deep learning models.

It offers:
- A flexible `AbstractAttention` interface.
- Implementations like `DotProductAttention`, `NNlibAttention`.
- A full `MultiHeadAttention` layer compatible with Flux.
- Utilities such as `make_causal_mask`.
- Support for custom Q/K transformations (e.g., for RoPE in `MultiHeadAttention`).
"""
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