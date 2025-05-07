"""
    AbstractAttention

Abstract type for attention mechanisms. Custom implementations should 
implement the `compute_attention` method.
"""
abstract type AbstractAttention end

"""
    compute_attention(mechanism::AbstractAttention, q, k, v, bias=nothing; 
                    mask=nothing, nheads=1, fdrop=identity)

Compute attention based on the specified mechanism.

# Arguments
- `mechanism`: The attention mechanism to use
- `q`: Query tensor of shape (d_model, seq_len_q, batch)
- `k`: Key tensor of shape (d_model, seq_len_k, batch)
- `v`: Value tensor of shape (d_model, seq_len_v, batch)
- `bias`: Optional bias tensor 
- `mask`: Optional mask tensor
- `nheads`: Number of attention heads
- `fdrop`: Dropout function to apply

# Returns
- `output`: Output tensor of shape (d_model, seq_len_q, batch)
- `attention_weights`: Attention weights
"""
function compute_attention end

"""
    DotProductAttention <: AbstractAttention

Standard scaled dot-product attention as described in "Attention is All You Need" paper.
"""
struct DotProductAttention <: AbstractAttention end

# Helper functions for multi-head attention
split_heads(x, nheads) = reshape(x, size(x, 1) ÷ nheads, nheads, size(x)[2:end]...)
join_heads(x) = reshape(x, :, size(x)[3:end]...)

function compute_attention(::DotProductAttention, q, k, v, bias=nothing; 
                          mask=nothing, nheads::Int=1, fdrop=identity)
    d_model, seq_len_q, batch_size = size(q)
    _, seq_len_k, _ = size(k)
    
    # Split heads using NNlib's approach
    q_4d = split_heads(q, nheads)  # (head_dim, nheads, seq_len_q, batch)
    k_4d = split_heads(k, nheads)  # (head_dim, nheads, seq_len_k, batch)
    v_4d = split_heads(v, nheads)  # (head_dim, nheads, seq_len_k, batch)
    
    # Compute attention scores
    kt = permutedims(k_4d, (3, 1, 2, 4))  # (seq_len_k, head_dim, nheads, batch)
    qt = permutedims(q_4d, (1, 3, 2, 4)) ./ sqrt(Float32(size(q_4d, 1)))  # (head_dim, seq_len_q, nheads, batch)
    scores = NNlib.batched_mul(kt, qt)  # (seq_len_k, seq_len_q, nheads, batch)
    
    # Apply bias if provided
    if bias !== nothing
        scores = scores .+ bias
    end
    
    # Apply mask if provided - exactly like NNlib does
    if mask !== nothing
        T = eltype(scores)
        scores = ifelse.(mask, scores, typemin(T))
    end
    
    # Apply softmax to get attention weights - after masking
    attn_weights = softmax(scores; dims=1)
    
    # Apply dropout if provided
    attn_weights = fdrop(attn_weights)
    
    # Apply attention weights to values
    vt = permutedims(v_4d, (1, 3, 2, 4))  # (head_dim, seq_len_k, nheads, batch)
    output = NNlib.batched_mul(vt, attn_weights)  # (head_dim, seq_len_q, nheads, batch)
    output = permutedims(output, (1, 3, 2, 4))  # (head_dim, nheads, seq_len_q, batch)
    
    # Join heads
    output = join_heads(output)
    
    # Return the attended output and the attention weights (post-softmax)
    return output, attn_weights
end

"""
    NNlibAttention <: AbstractAttention

Attention implementation that uses NNlib's dot_product_attention when available.
This provides a more optimized implementation that may be faster in some cases.
"""
struct NNlibAttention <: AbstractAttention end

function compute_attention(::NNlibAttention, q, k, v, bias=nothing; 
                          mask=nothing, nheads::Int=1, fdrop=identity)
    # Use NNlib's dot_product_attention implementation
    # This is more optimized and may be faster in some cases
    output, attention = NNlib.dot_product_attention(q, k, v, bias; 
                                                  mask=mask, nheads=nheads, fdrop=fdrop)
    
    # Return in our expected format
    # NNlib returns attention weights in (seq_len_k, seq_len_q, nheads, batch) format
    # which matches our format, so no need to permute
    return output, attention
end

"""
    make_causal_mask(x::AbstractArray, dims::Int=2)

Create a causal mask for a sequence of length derived from `x`.
The mask ensures that position `i` can only attend to positions `j ≤ i`.

# Arguments
- `x`: Input array from which sequence length is derived
- `dims`: Dimension along which to derive the sequence length (default: 2)

# Returns
- A boolean mask matrix of shape (seq_len, seq_len) where `true` indicates 
  allowed attention and `false` indicates masked (disallowed) attention.
"""
function make_causal_mask(x::AbstractArray, dims::Int=2)
    seq_len = size(x, dims)
    # Create boolean mask where true means "allow attention"
    return UpperTriangular(ones(Bool, seq_len, seq_len))
end 