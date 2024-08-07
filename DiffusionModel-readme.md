# Cat Diffusion Model Project

This project implements a simple Diffusion Model to generate images of cats. The model is trained on a dataset of cat images with corresponding facial landmark annotations.

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
- Visualization functions
- Model saving and loading functions
- Main execution script

## How It Works

1. The script loads a dataset of cat images and their facial landmark annotations.
2. It trains a simple Diffusion Model, which learns to reconstruct the input images.
3. The trained model can then generate new cat images by passing random noise through the model.

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
   - Display training loss curve
   - Visualize generated images
   - Save the trained model

4. To load and use the saved model:
   ```python
   loaded_model = DiffusionModel()
   loaded_model = load_model(loaded_model, "DiffusionModel")
   ```

## Model Architecture

The Diffusion Model uses a simple encoder-decoder architecture:
- Encoder: 3 convolutional layers with ReLU activations
- Decoder: 3 transposed convolutional layers with ReLU activations and a final Tanh activation

## Training Process

- The model is trained for a specified number of epochs (default: 50).
- In each iteration:
  1. The model tries to reconstruct the input images.
  2. MSE loss is calculated between the input and reconstructed images.
- Training loss is plotted after all epochs.

## Image Generation

The `visualize_generated_images` function shows the model's ability to reconstruct images:
1. It takes a batch of images from the dataset.
2. Passes them through the model.
3. Displays the original and reconstructed images side by side.

## Results

After training, you can:

1. Use the saved `DiffusionModel.pth` to generate or reconstruct cat images.
2. Experiment with different hyperparameters or architecture modifications to improve results.
3. Train for more epochs or on a larger dataset to potentially improve image quality.

Note: This implementation is a simplified version of a Diffusion Model and may not produce high-quality generated images. For better results, consider implementing a full Diffusion process with noise scheduling and sampling.
