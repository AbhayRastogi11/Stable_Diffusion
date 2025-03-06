using Flux
using Images
using ImageIO
using CUDA  # If using GPU
using Statistics

# Define image processing function
function load_and_preprocess_image(image_path)
    img = load(image_path)             # Load image
    img = imresize(img, (224, 224))    # Resize to 224x224
    img = Float32.(Gray.(img))         # Convert to Grayscale and Float32
    img = channelview(img)             # Ensure proper channel dimension
    return reshape(img, :, 1)          # Flatten into a vector
end

# Define a simple neural network
model = Chain(
    Dense(224*224, 128, relu),
    Dense(128, 64, relu),
    Dense(64, 10)  # Output 10 values (dummy classification task)
)

# Load a preprocessed image
image_path = "images/preprocessed_1.png"
input_tensor = load_and_preprocess_image(image_path)

# Check the shape to ensure correctness
println("Input tensor size: ", size(input_tensor))  # Should be (50176, 1)

# Perform a forward pass
output = model(input_tensor)
println("Model output: ", output)
