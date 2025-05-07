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

"""
    LinearAttention <: AbstractAttention

Linear Attention as described in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention".

# Fields
- `epsilon`: A small value added for numerical stability, particularly in the denominator of the attention calculation. Defaults to `1f-6`.
"""
struct LinearAttention{T<:Real} <: AbstractAttention
    epsilon::T
end

LinearAttention() = LinearAttention(1f-6)

# Feature map φ(x) = elu(x) + 1
phi(x) = NNlib.elu.(x) .+ 1

function compute_attention(mechanism::LinearAttention, q, k, v, bias=nothing; 
                          mask=nothing, nheads::Int=1, fdrop=identity)
    
    # Input shapes:
    # q: (d_model, seq_len_q, batch)
    # k: (d_model, seq_len_k, batch)
    # v: (d_model, seq_len_v, batch) - typically seq_len_k == seq_len_v
    # bias, mask: Accepted to match interface, but not used in this non-causal implementation.

    # Dimensions are implicitly handled by array operations below.
    # d_model, seq_len_q, batch_size = size(q)
    # _, seq_len_k, _ = size(k)

    # Split heads
    q_h = split_heads(q, nheads)  # (head_dim, nheads, seq_len_q, batch)
    k_h = split_heads(k, nheads)  # (head_dim, nheads, seq_len_k, batch)
    v_h = split_heads(v, nheads)  # (head_dim, nheads, seq_len_k, batch) (assuming head_dim_v = head_dim)

    # head_dim = size(q_h, 1) - also implicitly handled

    # Apply feature map φ
    phi_q = phi(q_h) # (head_dim, nheads, seq_len_q, batch)
    phi_k = phi(k_h) # (head_dim, nheads, seq_len_k, batch)

    # Non-Causal (Global) Linear Attention Computation
    
    # 1. Compute S_global = sum_j phi(K_j)V_j^T
    # phi_k is (head_dim, nheads, seq_len_k, batch)
    # v_h is (head_dim, nheads, seq_len_k, batch)
    # S_global result: (head_dim, head_dim, nheads, batch)
    
    # Permute for batched_mul:
    # phi_k_bm: (head_dim, seq_len_k, nheads, batch)
    # v_h_bm_T: (seq_len_k, head_dim, nheads, batch)
    phi_k_bm = permutedims(phi_k, (1,3,2,4)) 
    v_h_bm_T = permutedims(v_h, (3,1,2,4))
    
    S_global = NNlib.batched_mul(phi_k_bm, v_h_bm_T) # (head_dim, head_dim, nheads, batch)

    # 2. Compute Z_global = sum_j phi(K_j)
    # phi_k is (head_dim, nheads, seq_len_k, batch)
    # Z_global result: (head_dim, nheads, 1, batch)
    Z_global = sum(phi_k, dims=3) 

    # 3. Compute Numerator for all Q_i: Numerator_i = phi(Q_i)^T * S_global
    # phi_q is (head_dim, nheads, seq_len_q, batch)
    # S_global is (head_dim, head_dim, nheads, batch) -> permuted to (D_v, D_k, H, B)
    # phi_q is permuted to (D_k, T_q, H, B)
    # Numerator result: (head_dim_v, nheads, seq_len_q, batch)

    S_global_perm = permutedims(S_global, (2,1,3,4)) # (D_val, D_key, nheads, batch)
    phi_q_perm = permutedims(phi_q, (1,3,2,4))       # (D_key, seq_len_q, nheads, batch)
    
    numerator_h = NNlib.batched_mul(S_global_perm, phi_q_perm) # (D_val, seq_len_q, nheads, batch)
    numerator_h = permutedims(numerator_h, (1,3,2,4))          # (D_val, nheads, seq_len_q, batch)

    # 4. Compute Denominator for all Q_i: Denominator_i = phi(Q_i)^T * Z_global + epsilon
    # phi_q is (head_dim, nheads, seq_len_q, batch)
    # Z_global is (head_dim, nheads, 1, batch)
    # Denominator result: (1, nheads, seq_len_q, batch)
    denominator_h = sum(phi_q .* Z_global, dims=1) .+ mechanism.epsilon # Z_global broadcasts

    # 5. Compute per-head output
    output_h = numerator_h ./ denominator_h # (head_dim, nheads, seq_len_q, batch)

    # Apply dropout to the output
    output_h_after_dropout = fdrop(output_h)

    # Join heads
    output = join_heads(output_h_after_dropout) # (d_model, seq_len_q, batch)
    
    # Linear attention, in this formulation, does not produce a readily available QK^T style attention matrix.
    return output, nothing 
end
