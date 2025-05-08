# Attention.jl

Welcome to the documentation for `Attention.jl`.

This package provides modular and extensible attention mechanisms for deep learning models in Julia.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/mashu/Attention.jl.git")
```

## Main Features

- Modular attention mechanism interface
- `DotProductAttention`, `NNlibAttention`, `LinearAttention`, `MultiHeadAttention`
- Utilities like `make_causal_mask`
- Support for custom Q/K transformations (e.g., RoPE)

Check out the [API Reference](api/public.md) for detailed information on the functions and types provided. 