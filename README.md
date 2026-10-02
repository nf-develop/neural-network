Neural Network from scratch

A small neural network built from scratch using Python and NumPy, trained using the MNIST handwritten dataset.

Features:
- Loads and processes the MNIST dataset
- Normalises image pixel values
- Implements a neural network from scratch
- Uses ReLU and Softmax activation functions
- Implements forward propagation and backpropagation
- Uses gradient descent to train the network
- Predicts individual handwritten digits and displays the corresponding image

Network Structure:

784 input values
       ↓
10 hidden neurons
       ↓
10 output neurons
       ↓
Digits 0–9

Each image in the database is made up of 28x28 pixels, therefore 748 input values

Technologies:
- Python
- NumPy
- Pandas
- Matplotlib

The network achieved approximately 87% training accuracy after 950 iterations.

Usage:

Place mnist_train.csv in the project directory and run:

python neural_network.py

It should present iterations and accuracy of predictions. After training, user should be able to select a number index for an image within the database and the neural network will make a prediction based off that image