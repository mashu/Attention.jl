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

function compute_attention(::DotProductAttention, q, k, v, bias=nothing; 
                          mask=nothing, nheads::Int=1, fdrop=identity)
    d_model, seq_len_q, batch_size = size(q)
    _, seq_len_k, _ = size(k)
    
    # Calculate head dimensions
    head_dim = div(d_model, nheads)
    
    # Reshape to separate heads - (head_dim, num_heads, seq_len, batch)
    q_4d = reshape(q, head_dim, nheads, seq_len_q, batch_size)
    k_4d = reshape(k, head_dim, nheads, seq_len_k, batch_size)
    v_4d = reshape(v, head_dim, nheads, seq_len_k, batch_size)
    
    # Rearrange dimensions for batched_mul
    # We want (seq_len_q, head_dim) × (head_dim, seq_len_k) for each head and batch
    q_perm = permutedims(q_4d, (3, 1, 2, 4))  # (seq_len_q, head_dim, nheads, batch)
    k_perm = permutedims(k_4d, (1, 3, 2, 4))  # (head_dim, seq_len_k, nheads, batch)
    
    # Reshape for batched multiplication
    # Combine head and batch dimensions as the batch dimension for batched_mul
    q_reshaped = reshape(q_perm, seq_len_q, head_dim, nheads * batch_size)
    k_reshaped = reshape(k_perm, head_dim, seq_len_k, nheads * batch_size)
    
    # Compute attention scores
    scores = NNlib.batched_mul(q_reshaped, k_reshaped) ./ sqrt(Float32(head_dim))
    
    # Reshape scores back to (seq_len_q, seq_len_k, nheads, batch)
    scores_4d = reshape(scores, seq_len_q, seq_len_k, nheads, batch_size)
    
    # Apply mask if provided
    if mask !== nothing
        scores_4d = scores_4d .+ mask
    end
    
    # Apply bias if provided
    if bias !== nothing
        scores_4d = scores_4d .+ bias
    end
    
    # Apply softmax to get attention weights
    attn_weights = softmax(scores_4d; dims=2)
    
    # Apply dropout if provided
    attn_weights = fdrop(attn_weights)
    
    # Rearrange dimensions for value multiplication
    v_perm = permutedims(v_4d, (3, 1, 2, 4))  # (seq_len_k, head_dim, nheads, batch)
    v_reshaped = reshape(v_perm, seq_len_k, head_dim, nheads * batch_size)
    attn_reshaped = reshape(attn_weights, seq_len_q, seq_len_k, nheads * batch_size)
    
    # Apply attention weights to values
    output = NNlib.batched_mul(attn_reshaped, v_reshaped)
    
    # Reshape to final output shape (d_model, seq_len_q, batch)
    output = reshape(output, seq_len_q, head_dim, nheads, batch_size)
    output = permutedims(output, (2, 1, 3, 4))
    output = reshape(output, d_model, seq_len_q, batch_size)
    
    # Return the attended output and the attention scores
    return output, permutedims(attn_weights, (2, 1, 3, 4))
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
    # Create an upper triangular matrix including the diagonal
    # This is equivalent to a causal mask where each position can attend to itself and all previous positions
    return Bool.(UpperTriangular(ones(seq_len, seq_len)))
end 