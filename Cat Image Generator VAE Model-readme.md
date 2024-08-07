# VAE Cat Image Generator (Jupyter Notebook)

This project implements a Variational Autoencoder (VAE) to generate cat images using a single Jupyter notebook. The VAE is trained on a dataset of cat images and can then be used to generate new, synthetic cat images.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [Notebook Structure](#notebook-structure)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Image Generation](#image-generation)
8. [Usage](#usage)

## Project Overview

This project uses a Variational Autoencoder (VAE) to learn a latent representation of cat images and generate new, synthetic cat images. The VAE is implemented using PyTorch and trained on a custom dataset of cat images. The entire process, from data loading to model training and image generation, is contained within a single Jupyter notebook.

## Dependencies

- Python 3.x
- Jupyter Notebook
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- matplotlib
- numpy

## Dataset

The project uses a custom `CatDataset` class that loads cat images and their corresponding annotations from a specified directory. The dataset is expected to have the following structure:

```
root_directory/
    ├── image1.jpg
    ├── image1.jpg.cat
    ├── image2.jpg
    ├── image2.jpg.cat
    ...
```

Each `.jpg` file is an image of a cat, and the corresponding `.cat` file contains annotations for the cat's features.

## Notebook Structure

The Jupyter notebook is organized into several sections:

1. Imports and setup
2. Custom dataset definition
3. Data transformation and loading
4. Model definition
5. Loss function and optimizer setup
6. Training loop
7. Model saving and loading
8. Additional training (if needed)
9. Image generation

## Model Architecture

The VAE consists of an encoder and a decoder:

### Encoder
- 4 convolutional layers with ReLU activation
- Flattening layer
- Two fully connected layers for mean (μ) and log-variance (log σ²) of the latent space

### Decoder
- Fully connected layer to reshape the latent vector
- 4 transposed convolutional layers with ReLU activation (except the last layer, which uses Sigmoid)

## Training

The VAE is trained using the following process:

1. Load and preprocess the data using the custom `CatDataset` and `DataLoader`.
2. Define the VAE model, loss function, and optimizer.
3. Train the model for a specified number of epochs, computing both training and validation loss.
4. Plot the training and validation loss after each epoch.
5. Generate and display a sample image periodically during training.
6. Save the trained model.

## Image Generation

After training, the VAE can generate new cat images by:

1. Sampling a random vector from the latent space.
2. Passing this vector through the decoder part of the VAE.
3. Displaying the resulting image.

## Usage

1. Ensure all dependencies are installed in your Jupyter environment.
2. Open the Jupyter notebook in your Jupyter environment.
3. Update the `image_dir` variable to point to your dataset directory.
4. Run all cells in the notebook sequentially.
   - This will train the VAE on your dataset and save the model.
   - After training, it will generate and display a new cat image.
5. You can re-run the image generation cell multiple times to create different cat images.

Note: You may need to adjust hyperparameters like `latent_dim`, `num_epochs`, and learning rate as needed for your specific use case. These can be modified directly in the notebook cells.
