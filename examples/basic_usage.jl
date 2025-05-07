# Basic usage example for Attention.jl
using Attention
using Flux

# Create model parameters
d_model = 64    # Model dimension
nheads = 4      # Number of attention heads
seq_len = 10    # Sequence length
batch_size = 8  # Batch size

println("Creating a MultiHeadAttention layer with $d_model dimensions and $nheads heads")
mha = Attention.MultiHeadAttention(d_model, nheads, dropout_prob=0.1)

# Create random input data
println("Creating random input with shape ($d_model, $seq_len, $batch_size)")
x = rand(Float32, d_model, seq_len, batch_size)

# Apply self-attention
println("\nApplying self-attention")
output, attention = mha(x)

println("Output shape: ", size(output))
println("Attention shape: ", size(attention))

# Create a causal mask (useful for autoregressive models)
println("\nCreating a causal mask")
mask = Attention.make_causal_mask(x)
println("Mask shape: ", size(mask))

# Apply self-attention with causal masking
println("\nApplying self-attention with causal masking")
masked_output, masked_attention = mha(x, mask=mask)

println("Output shape: ", size(masked_output))
println("Attention shape: ", size(masked_attention))

# Check if the model is trainable
println("\nCreating a model with attention")

# Wrap MultiHeadAttention to only return the first element of the tuple
mha_layer = Chain(mha, first)

# Now we can use it in a larger model
model = Chain(
    Dense(d_model => d_model),
    mha_layer,
    Dense(d_model => d_model)
)

# Now the Chain will work with the wrapped attention layer
println("\nApplying the model to the input")
model_output = model(x)
println("Model output shape: ", size(model_output))

# You can also use different attention mechanisms
println("\nCreating another attention layer with a different mechanism")
mha2 = Attention.MultiHeadAttention(d_model, nheads, attention_impl=Attention.NNlibAttention())
output2, attention2 = mha2(x)

println("Output shape: ", size(output2))
println("Attention shape: ", size(attention2)) 