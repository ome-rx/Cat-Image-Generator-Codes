# Cat Diffusion Model Project

This project implements a Diffusion Model to generate images of cats. The model is trained on a dataset of cat images with corresponding facial landmark annotations.

## Table of Contents

1. [Requirements](#requirements)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Training Process](#training-process)
7. [Image Generation](#image-generation)
8. [Results](#results)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Pillow (PIL)
- matplotlib
- numpy
- tqdm

You can install the required packages using pip:

```
pip install torch torchvision pillow matplotlib numpy tqdm
```

## Project Structure

The project consists of a single Python script with the following main components:

- CatDataset class
- DiffusionModel class
- Training function
- Image generation function
- Main execution script

## How It Works

1. The script loads a dataset of cat images and their facial landmark annotations.
2. It trains a Diffusion Model, which learns to gradually denoise images.
3. The trained model can then generate new cat images by reversing the diffusion process.

## Usage

1. Prepare your dataset:
   - Organize your cat images in a directory.
   - Each image should have a corresponding annotation file (`.cat`) with facial landmarks.
   - Update the `image_dir` variable to point to your dataset.

2. Run the script:
   ```
   python cat_diffusion.py
   ```

3. The script will:
   - Train the Diffusion Model
   - Display training and validation loss curves
   - Generate a sample image
   - Save the trained model

4. To load and use the saved model:
   ```python
   model = DiffusionModel()
   model.load_state_dict(torch.load("CatDiffusionModel.pth"))
   generate_image(model)
   ```

## Model Architecture

The Diffusion Model uses a U-Net architecture:
- Input: Noisy image (3 channels) and time step (3 channels, repeated)
- Output: Predicted noise (3 channels)
- Architecture: 4 downsampling layers and 3 upsampling layers with skip connections

## Training Process

- The model is trained for a specified number of epochs (default: 5).
- In each iteration:
  1. Random timesteps are selected for each image in the batch.
  2. Noise is added to the images based on the timesteps.
  3. The model predicts the added noise.
  4. MSE loss is calculated between predicted and actual noise.
- Training and validation losses are plotted after each epoch.

## Image Generation

The `generate_image` function implements the reverse diffusion process:
1. Start with random noise.
2. Iteratively denoise the image using the trained model.
3. The final result is a generated cat image.

## Results

The quality of generated images should improve over time. After training, you can:

1. Use the saved `CatDiffusionModel.pth` to generate new cat images.
2. Experiment with different hyperparameters or architecture modifications to improve results.
3. Train for more epochs or on a larger dataset to potentially improve image quality.

Note: Diffusion Models can be computationally intensive to train and generate images. You might need a GPU for faster processing, especially with larger image sizes or more diffusion steps.
