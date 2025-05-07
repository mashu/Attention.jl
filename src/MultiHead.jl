"""
    MultiHeadAttention(d_model, nheads=8; bias=false, dropout_prob=0.0, attention_impl=DotProductAttention())

The multi-head dot-product attention layer used in Transformer architectures.

Returns the transformed input sequence and the attention scores.

# Arguments

- `d_model`: The embedding dimension
- `nheads`: number of heads. Default `8`.
- `bias`: whether pointwise QKVO dense transforms use bias. Default `false`.
- `dropout_prob`: dropout probability for the attention scores. Default `0.0`.
- `attention_impl`: the attention implementation to use. Default `DotProductAttention()`.
"""
struct MultiHeadAttention{P1, D, P2, A<:AbstractAttention}
  nheads::Int
  head_size::Int
  q_proj::P1
  k_proj::P1
  v_proj::P1
  attn_drop::D
  out_proj::P2
  attn_impl::A
end

Flux.@layer MultiHeadAttention

function MultiHeadAttention(d_model::Int, nheads::Int=8; 
                     bias::Bool=false,
                     dropout_prob=0.0,
                     attention_impl=DotProductAttention())

  d_model % nheads == 0 || throw(ArgumentError("d_model ($d_model) must be divisible by nheads ($nheads)"))
  
  head_size = div(d_model, nheads)
  
  q_proj = Dense(d_model => d_model; bias=bias)
  k_proj = Dense(d_model => d_model; bias=bias)
  v_proj = Dense(d_model => d_model; bias=bias)
  attn_drop = Dropout(dropout_prob)
  out_proj = Dense(d_model => d_model; bias=bias)
  
  return MultiHeadAttention(nheads, head_size, q_proj, k_proj, v_proj, attn_drop, out_proj, attention_impl)
end

# self-attention
(mha::MultiHeadAttention)(qkv; kws...) = mha(qkv, qkv, qkv; kws...)

# key and value are the same
(mha::MultiHeadAttention)(q, kv; kws...) = mha(q, kv, kv; kws...)

function (mha::MultiHeadAttention)(q_in, k_in, v_in, bias=nothing; mask=nothing)
  ## [q_in] = [q_in_dim, q_len, batch_size]
  ## [k_in] = [k_in_dim, kv_len, batch_size] 
  ## [v_in] = [v_in_dim, kv_len, batch_size]
  q = mha.q_proj(q_in)  # [q] = [qk_dim, q_len, batch_size]
  k = mha.k_proj(k_in)  # [k] = [qk_dim, kv_len, batch_size] 
  v = mha.v_proj(v_in)  # [v] = [v_dim, kv_len, batch_size]
  
  # Use the attention implementation
  x, α = compute_attention(mha.attn_impl, q, k, v, bias; 
                           mask=mask, nheads=mha.nheads, fdrop=mha.attn_drop)
  
  x = mha.out_proj(x)
  # [x] = [out_dim, q_len, batch_size]
  # [α] = [kv_len, q_len, nheads, batch_size]
  return x, α
end 