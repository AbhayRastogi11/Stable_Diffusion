import torch
from diffusers import StableDiffusionPipeline
import os

# Define output directory
output_dir = "../images"
os.makedirs(output_dir, exist_ok=True)

# Define the text prompt
prompt = "a serene sunset over a futuristic city"

# Load the Stable Diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# Generate images
num_images = 3
for i in range(num_images):
    image = pipeline(prompt).images[0]  # Generate image
    image_path = os.path.join(output_dir, f"generated_{i+1}.png")
    image.save(image_path)
    print(f"Image saved: {image_path}")

print("Image generation completed.")