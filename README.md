# Fetal Biometry prediction

## Introduction
This project implements the BiometryDetection model, a deep learning architecture for detecting biometric features in images. The model is designed for classification tasks, where the goal is to predict class labels for the input images. It takes grayscale images as input (with 1 channel) and outputs class probabilities for 8 distinct classes.

The architecture consists of two convolutional layers followed by max pooling, dropout for regularization, and a fully connected layer for classification. Max pooling is utilized to downsample the feature maps, and dropout is applied after the second convolutional layer to mitigate overfitting. The forward method defines the forward pass of the network, processing the input through the convolutional layers and ultimately producing class predictions.

## Project Structure
The project is organized as follows:

Project/ 
 
    ├── model.py # Contains the implementation of the SegNet model 
    ├── train.py # Script to train the model on a dataset 
    ├── dataset.py # Dataset loader for input data 
    ├── utils.py # Utility functions such as saving/loading models 
    ├── requirements.txt # Python dependencies 
    ├── README.md # Project description 
    └── data/ # Directory for storing datasets (e.g., images and labels)



### 1. model.py
This file contains the implementation of the BiometryDetection model using PyTorch. The network consists of two convolutional layers, followed by max pooling, dropout, and a fully connected layer for classification. The architecture is as follows:

**Convolutional Layers:**
- **Layer 1:** A convolutional layer with 12 filters, kernel size 3, and padding 1, followed by ReLU activation and max pooling.
- **Layer 2:** A convolutional layer with 24 filters, kernel size 3, and padding 1, followed by ReLU activation and max pooling.

**Regularization:**
- **Dropout:** Applied after the second convolutional layer to prevent overfitting, with a dropout probability of 0.2.

**Fully Connected Layer:**
- A linear layer that takes the flattened output from the convolutional layers and predicts class probabilities for 8 biometry points.

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
