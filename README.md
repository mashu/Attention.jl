# Attention

[![Build Status](https://github.com/mashu/Attention.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mashu/Attention.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mashu/Attention.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mashu/Attention.jl)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](Project.toml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://mashu.github.io/Attention.jl/dev/)

A Julia package providing modular and extensible attention mechanisms for deep learning models.

## Features

- Modular attention mechanism interface through the `AbstractAttention` type
- Ready-to-use implementations including:
  - `DotProductAttention`: Standard scaled dot-product attention
  - `NNlibAttention`: Wrapper for `NNlib.dot_product_attention`, allowing use of optimized kernels where available.
  - `LinearAttention`: Linear complexity attention ("Transformers are RNNs")
  - `MultiHeadAttention`: Full implementation of multi-head attention compatible with Flux
- Useful utilities like `make_causal_mask` for creating causal masks
- Support for custom transformations on Q and K tensors in `MultiHeadAttention` (e.g., for RoPE)
- Fully compatible with automatic differentiation frameworks (Zygote, CUDA)
- Clean, efficient implementation with minimal dependencies

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/mashu/Attention.jl.git")
```

## Usage

```julia
using Attention
using Flux

# Create a MultiHeadAttention layer
mha = Attention.MultiHeadAttention(512, 8, dropout_prob=0.1)

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

### Applying Transformations to Queries and Keys

`MultiHeadAttention` supports applying custom transformations to the query (Q) and key (K) tensors. This is done *after* their initial linear projections but *before* the attention scores are computed. This feature is useful for incorporating advanced positional embedding techniques, such as Rotary Positional Embeddings (RoPE).

To use this, provide functions to the `q_transform` and `k_transform` keyword arguments in the `MultiHeadAttention` constructor. Both default to `identity` (no transformation). For RoPE, you would typically pass the same RoPE transformation function to both arguments.

### Specifying the Underlying Attention Mechanism

By default, `MultiHeadAttention` uses `DotProductAttention` as its core attention calculation method. However, you can specify a different underlying attention mechanism by passing an instance of any `AbstractAttention` subtype to the `attention_impl` keyword argument.

This allows you to easily swap out attention implementations, for example, to use optimized versions or experiment with custom behaviors, as long as they conform to the `AbstractAttention` interface (primarily by implementing the `Attention.compute_attention` method).

For example, to explicitly use `NNlibAttention` (which wraps `NNlib.dot_product_attention` and can leverage optimized kernels):

```julia
using Attention
using Flux

# Example input (same as above)
batch_size = 32
seq_len = 20
d_model = 512
x = rand(Float32, d_model, seq_len, batch_size)

# Using NNlibAttention explicitly within MultiHeadAttention
mha_with_nnlib = Attention.MultiHeadAttention(d_model, 8, attention_impl=Attention.NNlibAttention())

# This instance of MultiHeadAttention will use NNlib.dot_product_attention for its core calculations.
# output, attention_weights = mha_with_nnlib(x)

# If you had defined your own mechanism, e.g.:
# struct MyCustomMechanism <: AbstractAttention end
# function Attention.compute_attention(::MyCustomMechanism, q, k, v, ...)
#   # ... your implementation ...
# end
# You would pass it as:
# mha_custom = Attention.MultiHeadAttention(d_model, 8, attention_impl=MyCustomMechanism())
```

## Examples

### MNIST Image Classification with Attention

The `examples/mnist_classification/` directory contains a complete example of using `Attention.jl` to build a model for classifying MNIST digits. This example demonstrates:

-   Integrating `MultiHeadAttention` with a Convolutional Neural Network (CNN) front-end.
-   Preprocessing image data to make it suitable for attention layers (reshaping feature maps into sequences).
-   A full training and evaluation loop using Flux.jl and MLDatasets.jl, showcasing modern Flux training style with `Flux.setup` and `Flux.withgradient`.

To run the example:
1.  Navigate to the `examples/mnist_classification/` directory in your terminal.
2.  Activate the project environment: `julia --project=.`
3.  Instantiate the project dependencies by running the following in the Julia REPL:
    ```julia
    using Pkg; Pkg.instantiate()
    ```
4.  Execute the training script from your terminal: `julia train_mnist.jl`

This example provides a practical guide for incorporating attention mechanisms from this package into larger deep learning models.

## Dimensions

Throughout this package, arrays follow the Julia/Flux convention with dimensions:
- Features/embedding first: (d_model, seq_len, batch)
- For attention weights: (seq_len_k, seq_len_q, num_heads, batch)

## License

MIT License 
