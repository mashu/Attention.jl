# Attention.jl

A Julia package providing modular and extensible attention mechanisms for deep learning models.

## Features

- Modular attention mechanism interface through the `AbstractAttention` type
- Ready-to-use implementations including:
  - `DotProductAttention`: Standard scaled dot-product attention
  - `NNlibAttention`: Interface to NNlib's attention (currently a fallback to `DotProductAttention`)
- `MultiHeadAttention`: Full implementation of multi-head attention compatible with Flux
- Useful utilities like `make_causal_mask` for creating causal masks
- Fully compatible with automatic differentiation frameworks (Zygote, CUDA)
- Clean, efficient implementation with minimal dependencies

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/username/Attention.jl.git")
```

## Usage

```julia
using Attention
using Flux

# Create a MultiHeadAttention layer
mha = MultiHeadAttention(512, 8, dropout_prob=0.1)

# Create sample input
batch_size = 32
seq_len = 20
d_model = 512
x = rand(Float32, d_model, seq_len, batch_size)

# Self-attention
output, attention = mha(x)

# Create a causal mask (for autoregressive models)
mask = make_causal_mask(x)

# Apply attention with mask
output, attention = mha(x, mask=mask)
```

### Custom Attention Mechanisms

You can create custom attention mechanisms by extending the `AbstractAttention` type:

```julia
struct MyCustomAttention <: AbstractAttention
    # Custom parameters here
end

function Attention.compute_attention(::MyCustomAttention, q, k, v, bias=nothing; 
                                   mask=nothing, nheads=1, fdrop=identity)
    # Custom implementation here
    return output, attention_weights
end

# Use with MultiHeadAttention
mha = MultiHeadAttention(512, 8, attention_impl=MyCustomAttention())
```

## Dimensions

Throughout this package, arrays follow the Julia/Flux convention with dimensions:
- Features/embedding first: (d_model, seq_len, batch)
- For attention weights: (seq_len_k, seq_len_q, num_heads, batch)

## License

MIT License 