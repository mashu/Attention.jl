using Flux
using MLDatasets
using Attention # Assuming this is accessible, e.g., from the Project.toml path
using Statistics
using CUDA # For GPU support, if available and enabled

# --- Helper Functions ---
function get_data(batch_size)
    # Load MNIST using the new API
    xtrain, ytrain = MLDatasets.MNIST(Float32, split=:train)[:]
    xtest, ytest = MLDatasets.MNIST(Float32, split=:test)[:]

    # Reshape and batch
    # MNIST images are 28x28. We add a channel dimension (1) and batch dimension.
    # Output shape: (width, height, channels, batch_size)
    xtrain = Flux.unsqueeze(xtrain, 3) # Add channel dimension
    xtest = Flux.unsqueeze(xtest, 3)   # Add channel dimension

    # One-hot encode labels
    ytrain = Flux.onehotbatch(ytrain, 0:9)
    ytest = Flux.onehotbatch(ytest, 0:9)

    # Create DataLoaders
    train_loader = Flux.DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader((xtest, ytest), batchsize=batch_size)

    return train_loader, test_loader
end

function accuracy(data_loader, model, device)
    acc = 0
    num = 0
    for (x, y) in data_loader
        x, y = x |> device, y |> device
        acc += sum(Flux.onecold(model(x)) .== Flux.onecold(y))
        num += size(y, 2)
    end
    return acc / num
end

# --- Model Definition ---
function create_model(d_model::Int, nheads::Int, device)
    # MNIST images are 28x28 pixels.
    # After Conv layers, we'll have a feature map that we can flatten and treat as a sequence.
    # For example, two Conv layers might reduce 28x28 to 7x7.
    # The number of channels from the last Conv layer will be `d_model` for the attention layer.
    # The sequence length will be 7*7 = 49.

    # Simplified CNN front-end
    cnn_feature_extractor = Chain(
        Conv((3, 3), 1 => 16, relu, pad=SamePad()), # Output: 28x28x16
        MaxPool((2, 2)),                            # Output: 14x14x16
        Conv((3, 3), 16 => d_model, relu, pad=SamePad()),# Output: 14x14x_d_model_
        MaxPool((2, 2))                             # Output: 7x7x_d_model_
    ) |> device

    # Attention layer expects input of shape (features, seq_len, batch_size)
    # `d_model` is the feature dimension from the CNN
    # `seq_len` will be 7*7 = 49 after flattening the spatial dimensions of the CNN output
    mha = Attention.MultiHeadAttention(d_model, nheads) |> device

    # Classifier head
    classifier_head = Dense(d_model, 10) |> device # 10 classes for MNIST

    return Chain(
        cnn_feature_extractor,
        # Reshape: (width, height, channels, batch) -> (channels, width*height, batch)
        # This is (d_model, seq_len, batch_size) for the attention layer
        x -> reshape(x, size(x, 3), size(x, 1) * size(x, 2), size(x, 4)),
        # Input to MHA: (d_model, seq_len, batch)
        # Output from MHA: (output_features, seq_len, batch), (attention_weights)
        # We take the first element (output_features)
        (x) -> mha(x)[1],
        # Global Average Pooling across the sequence dimension (seq_len)
        # Input: (d_model, seq_len, batch) -> Output: (d_model, 1, batch)
        x -> mean(x, dims=2),
        Flux.flatten, # (d_model, 1, batch) -> (d_model, batch)
        classifier_head
    ) |> device
end

# --- Training Setup ---
function train(; epochs=10, batch_size=128, lr=3e-4, d_model=32, nheads=4)
    # Check for GPU availability and use it if possible
    if CUDA.functional()
        @info "Training on CUDA GPU"
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Get Data
    train_loader, test_loader = get_data(batch_size)

    # Create Model
    model = create_model(d_model, nheads, device)

    # Optimizer
    opt_state = Flux.setup(Flux.Adam(lr), model)

    # Loss function
    # Renamed to avoid conflict if a different `loss` variable is needed elsewhere
    # This function now takes the model as an argument
    loss_fn(m, x_batch, y_batch) = Flux.logitcrossentropy(m(x_batch), y_batch)

    # Training Loop
    @info "Starting training..."
    for epoch in 1:epochs
        for (x_batch, y_batch) in train_loader
            x_dev, y_dev = x_batch |> device, y_batch |> device
            # Calculate gradient and loss value
            val, grads = Flux.withgradient(model) do m
                loss_fn(m, x_dev, y_dev)
            end
            Flux.update!(opt_state, model, grads[1])
        end
        
        # Log progress
        train_acc = accuracy(train_loader, model, device)
        test_acc = accuracy(test_loader, model, device)
        # Calculate loss on the first batch of the test set for logging
        sample_x_test, sample_y_test = first(test_loader)
        current_epoch_loss = loss_fn(model, sample_x_test |> device, sample_y_test |> device)
        println("Epoch: $epoch, Loss: $current_epoch_loss, Train Acc: $(round(train_acc*100, digits=2))%, Test Acc: $(round(test_acc*100, digits=2))%")
    end

    @info "Training complete."
    test_acc = accuracy(test_loader, model, device)
    @info "Final test accuracy: $(round(test_acc*100, digits=2))%"

    return model
end

# --- Run Training ---
if abspath(PROGRAM_FILE) == @__FILE__
    # Hyperparameters
    const D_MODEL = 32       # Dimension for attention (must be divisible by nheads)
    const N_HEADS = 4        # Number of attention heads
    const BATCH_SIZE = 256
    const LEARNING_RATE = 1e-3 # Adjusted from 3e-4 for potentially faster initial convergence
    const EPOCHS = 5         # Reduced for a quick example run

    trained_model = train(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        d_model=D_MODEL,
        nheads=N_HEADS
    )
    # You can save the model or do further evaluation here
    # For example: using BSON: @save "mnist_attention_model.bson" trained_model
end 