# 🚀 Synthetic Image Generation, Preprocessing, and Flux Model Forward Pass

## 📌 Project Overview

This project demonstrates the following key tasks:

- **Synthetic Image Generation**: Using Stable Diffusion (via Hugging Face's `diffusers` library) to generate images from text prompts.
- **Image Preprocessing**: Resizing, normalizing, and preparing images for deep learning models.
- **Flux-Based Neural Network Forward Pass**: Implementing a minimal neural network in Flux (Julia) and running a forward pass on preprocessed images.

### 📊 Pipeline Overview:
- **Python**: Image generation and preprocessing
- **Julia**: Neural network implementation and execution

---

## 📂 Project Structure

```
Synthetic_Image_Flux/
├── images/                    # Stores generated and preprocessed images
│   ├── generated_1.png        # Generated images from Stable Diffusion
│   ├── generated_2.png
│   ├── generated_3.png
│   ├── preprocessed_1.png     # Preprocessed images (224x224, normalized)
│   ├── preprocessed_2.png
│   ├── preprocessed_3.png
│   ├── forward_pass_output_3.png   # Result of forward pass in Flux model
│
├── src/                       # Source code directory
│   ├── image_generation.py    # Python script for Stable Diffusion image generation
│   ├── preprocessing.py       # Python script for image preprocessing
│   ├── flux_model.jl          # Julia script for Flux model and forward pass
│
├── notebooks/                 # Jupyter notebooks for step-by-step execution
│   ├── image_generation.ipynb # Notebook for generating images
│   ├── preprocessing.ipynb    # Notebook for preprocessing images
│
├── requirements.txt           # Python dependencies (diffusers, OpenCV, etc.)
├── Project.toml               # Julia dependencies for Flux
└── README.md                  # This documentation file
```

---

## 💻 Setup & Installation

### 🔹 Python Environment Setup

Ensure Python 3.8+ is installed on your system.

(Optional) Create a virtual environment for the project:
```bash
python -m venv env
```
Activate the environment:
```bash
# On Linux/Mac:
source env/bin/activate
# On Windows:
env\Scripts\activate
```
Install required Python packages:
```bash
pip install -r requirements.txt
```

### 🔹 Julia Environment Setup

Install Julia from the [official website](https://julialang.org/downloads/).

Set up Julia dependencies:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

---

## 📜 Execution Guide

### 1️⃣ Generating Synthetic Images

You can generate images using either the script or the Jupyter Notebook:

- Using the script:
  ```bash
  python src/image_generation.py
  ```
- Using the Jupyter Notebook:
  ```bash
  jupyter notebook notebooks/image_generation.ipynb
  ```

✅ **Expected Output:**
- Three images (`generated_1.png`, `generated_2.png`, `generated_3.png`) will be saved in the `images/` directory.

---

### 2️⃣ Preprocessing Images

To resize and normalize the generated images:

- Using the script:
  ```bash
  python src/preprocessing.py
  ```
- Using the Jupyter Notebook:
  ```bash
  jupyter notebook notebooks/preprocessing.ipynb
  ```

✅ **Expected Output:**
- Preprocessed images (`preprocessed_1.png`, `preprocessed_2.png`, `preprocessed_3.png`) will be saved in the `images/` directory.

---

### 3️⃣ Running Flux Model in Julia

To execute the Flux-based neural network forward pass:

1. Open Julia in the project directory:
   ```bash
   julia
   ```
2. Run the Julia script:
   ```julia
   include("src/flux_model.jl")
   ```

✅ **Expected Output:**
- A tensor output representing the processed image will be displayed.

---

## 🛠 Troubleshooting

### 🔹 Python Issues

- If `diffusers` fails to install:
  ```bash
  pip install diffusers --upgrade
  ```
- If CUDA is not available for GPU acceleration, ensure you have a compatible GPU and install the appropriate PyTorch version:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

### 🔹 Julia Issues

- If `Flux` or related packages are missing, manually add them:
  ```julia
  using Pkg
  Pkg.add("Flux")
  Pkg.add("Images")
  Pkg.add("ImageIO")
  Pkg.add("CUDA")
  ```

---

## 📚 Additional Notes

- **Model Customization**: Modify the `flux_model.jl` script to add layers or adjust the neural network architecture.
- **Stable Diffusion Customization**: Update the text prompt in `image_generation.py` to generate different image styles.
- **GPU Support**: Ensure both Python and Julia environments are configured to utilize GPU acceleration via CUDA.

---

## 📌 Author & Credits

- **Developed by**: Abhay Rastogi

### 🧰 Dependencies:
- **Stable Diffusion** (via Hugging Face's `diffusers`)
- **Flux.jl** (for machine learning in Julia)
- **OpenCV & NumPy** (for image processing)

If you have any questions or issues, feel free to reach out! 🚀