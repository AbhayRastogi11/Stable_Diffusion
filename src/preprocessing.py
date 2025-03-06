import os
import cv2
import numpy as np
from glob import glob

# Define input and output directories
input_dir = "../images"
output_dir = "../images"
os.makedirs(output_dir, exist_ok=True)

# Target size for resizing
image_size = (224, 224)

# Function to preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)  # Resize to 224x224
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    return image

# Process all generated images
image_paths = glob(os.path.join(input_dir, "generated_*.png"))

for idx, image_path in enumerate(image_paths):
    processed_image = preprocess_image(image_path)
    output_path = os.path.join(output_dir, f"preprocessed_{idx+1}.png")
    cv2.imwrite(output_path, (processed_image * 255).astype(np.uint8))  # Save as PNG
    print(f"Processed image saved: {output_path}")

print("Image preprocessing completed.")