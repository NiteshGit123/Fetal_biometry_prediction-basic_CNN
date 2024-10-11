# SegNet Image Segmentation

## Introduction
This project implements a simplified version of SegNet, a deep convolutional encoder-decoder architecture for image segmentation. SegNet is designed for pixel-wise classification, making it suitable for tasks like semantic segmentation, where the goal is to assign a class label to each pixel in the input image. The network in this project takes grayscale images as input (with 1 channel) and outputs a single-channel segmented image (binary or multi-class).

The architecture consists of three encoding and three decoding stages, each having convolutional and batch normalization layers. Max pooling is used during the encoding phase to downsample the feature maps, and indices of pooling are saved to enable unpooling in the decoding phase.

## Project Structure
The project is organized as follows:

SegNet-Project/ 
├── model.py # Contains the implementation of the SegNet model 
├── train.py # Script to train the model on a dataset 
├── dataset.py # Dataset loader for input data 
├── utils.py # Utility functions such as saving/loading models 
├── requirements.txt # Python dependencies 
├── README.md # Project description 
└── data/ # Directory for storing datasets (e.g., images and labels)



### 1. model.py
This file contains the implementation of the SegNet architecture using PyTorch. The network uses 3 stages of encoding and decoding layers. The architecture is as follows:

**Encoding:**
- **Stage 1:** Two convolutional layers with 64 filters each, followed by batch normalization and max pooling with indices.
- **Stage 2:** Two convolutional layers with 128 filters each, followed by batch normalization and max pooling with indices.
- **Stage 3:** Three convolutional layers with 256 filters each, followed by batch normalization and max pooling with indices.

**Decoding:**
- **Stage 3:** Three convolutional layers with 256 filters each, followed by batch normalization and unpooling.
- **Stage 2:** Two convolutional layers with 128 filters each, followed by batch normalization and unpooling.
- **Stage 1:** Two convolutional layers with 64 filters each, followed by batch normalization and unpooling to recover the input spatial size.

The forward method defines the forward pass of the network.

### 2. train.py
This script is responsible for training the SegNet model on a custom dataset. It uses PyTorch's DataLoader to load the images and corresponding segmentation masks, computes the loss, and performs backpropagation to update the model's weights.

**Key features:**
- Training loop to optimize the model using a loss function such as cross-entropy loss for segmentation tasks.
- Periodic saving of model checkpoints.
- Option to visualize or log training metrics like loss and accuracy.

### 3. dataset.py
This script defines a custom PyTorch dataset class for loading images and their corresponding segmentation masks. It is assumed that the data is stored in two directories, one for images and one for labels.

The dataset class handles:
- Loading and preprocessing the images and labels.
- Performing transformations such as normalization, resizing, and data augmentation.

### 4. utils.py
This script contains utility functions used throughout the project, such as:
- Functions to save and load model checkpoints.
- Functions for visualization of the results (e.g., comparing predicted segmentation masks with the ground truth).

### 5. requirements.txt
This file lists all the Python dependencies required to run the project. To install the dependencies, run:

pip install -r requirements.txt


The data directory should contain the images and corresponding segmentation masks used for training and validation. The images should be grayscale images, and the labels should be binary masks (for binary classification) or have multiple channels (for multi-class segmentation).

data/
├── train/
   ├── images/       # Training images
   └── masks/        # Corresponding segmentation masks
├── val/
   ├── images/       # Validation images
   └── masks/        # Corresponding segmentation masks
├── test/
    ├── images/       # Testing images
    └── masks/        # Corresponding segmentation masks
