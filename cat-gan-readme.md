# Cat GAN Project

This project implements a Generative Adversarial Network (GAN) to generate images of cats. The GAN is trained on a dataset of cat images and can generate new, synthetic cat images.

## Table of Contents

1. [Requirements](#requirements)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Training Process](#training-process)
7. [Results](#results)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Pillow (PIL)
- matplotlib
- numpy

You can install the required packages using pip:

```
pip install torch torchvision pillow matplotlib numpy
```

## Project Structure

The project consists of a single Python script with the following main components:

- Generator class
- Discriminator class
- CatDataset class
- Training function
- Main execution script

## How It Works

1. The script loads a dataset of cat images.
2. It trains a GAN, which consists of two neural networks:
   - A Generator that creates synthetic cat images from random noise.
   - A Discriminator that tries to distinguish real cat images from generated ones.
3. Through adversarial training, the Generator learns to create increasingly realistic cat images.

## Usage

1. Prepare your dataset:
   - Organize your cat images in subdirectories within a main directory.
   - Update the `root_dir` in the main execution section to point to your dataset.

2. Run the script:
   ```
   python cat_gan.py
   ```

3. The script will train the GAN and periodically display progress:
   - Loss curves for the Generator and Discriminator
   - Sample generated images

4. After training, the models are saved as `generator.pth` and `discriminator.pth`.

5. The script generates a final set of images using the trained Generator.

## Model Architecture

### Generator
- Input: Random noise vector (latent_dim = 100)
- Output: 64x64 RGB image
- Architecture: 5 transposed convolutional layers with batch normalization and ReLU activations

### Discriminator
- Input: 64x64 RGB image
- Output: Scalar value (probability of input being real)
- Architecture: 5 convolutional layers with batch normalization and LeakyReLU activations

## Training Process

- The GAN is trained for a specified number of epochs (default: 200).
- In each iteration:
  1. The Generator creates fake images from random noise.
  2. The Discriminator is trained to distinguish real from fake images.
  3. The Generator is trained to fool the Discriminator.
- Loss curves and sample generated images are displayed every 50 batches.

## Results

The quality of generated images should improve over time. After training, you can:

1. Use the saved `generator.pth` to generate new cat images.
2. Experiment with different hyperparameters or architecture modifications to improve results.
3. Use the trained model as a starting point for transfer learning on similar tasks.

Note: GANs can be challenging to train, and results may vary. You might need to adjust hyperparameters or train for more epochs to achieve satisfactory results.
