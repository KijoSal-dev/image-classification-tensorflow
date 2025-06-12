# image-classification-tensorflow
# MNIST Image Classification with TensorFlow

This notebook demonstrates building and training a simple neural network using TensorFlow and Keras to classify images from the MNIST dataset.

## Dataset

The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems. It consists of 60,000 training images and 10,000 testing images.

## Model Architecture

The model is a sequential neural network with the following layers:

1.  **Flatten**: Reshapes the 28x28 pixel images into a 1D array of 784 pixels.
2.  **Dense (128 neurons, ReLU activation)**: A fully connected layer with 128 neurons and the Rectified Linear Unit (ReLU) activation function.
3.  **Dropout (0.2)**: Randomly sets 20% of the input units to 0 during training to prevent overfitting.
4.  **Dense (10 neurons, Softmax activation)**: A fully connected layer with 10 neurons (one for each digit) and the Softmax activation function to output a probability distribution over the classes.

## Training

The model is compiled with the Adam optimizer and `sparse_categorical_crossentropy` loss function. It is trained for 5 epochs on the training data.

## Evaluation

The model is evaluated on the test data to assess its performance on unseen images. The output of the `evaluate` method shows the loss and accuracy on the test set.

## Code

The code in this notebook performs the following steps:

1.  Imports necessary libraries (TensorFlow).
2.  Loads and preprocesses the MNIST dataset (normalization).
3.  Defines the sequential neural network model.
4.  Compiles the model with an optimizer, loss function, and metrics.
5.  Trains the model on the training data.
6.  Evaluates the trained model on the test data.
