import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#load MNIST database
data = pd.read_csv("mnist_train.csv")
data.head()
data = np.array(data)
m, n = data.shape 
np.random.shuffle(data)

#seperate the first 1000 images for development
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

#remaining for training
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

def init_params(): #initialising weights and biases
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z): #ReLU activation function
    return np.maximum(0, Z)

def softmax(Z): #convert output values into probabilities
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)


def forward_prop(W1, b1, W2, b2, X): #pass the input through the network
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y): #convert the correct labels into one-hot coded vectors
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z): #derivative of the ReLU function
    return Z > 0


def back_prop(Z1, A1, Z2, A2, W2, X, Y): #calculate the gradients used to improve the network
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): #update the weights and biases using calculated gradients
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1 
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2): #receive the predicted digit from the network's output
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y): #calculate the accuracy of the predictions
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha): #train neural network using gradient descent
    W1, b1, W2, b2 = init_params()
    for x in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (x % 50 == 0):
            print("Iteration: ", x)
            print("Accuracy:", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1) #training network

def make_prediction(index, X, Y): #making prediction on an individual image
    prediction = get_predictions(forward_prop(W1, b1, W2, b2, X[:, index:index + 1])[3])
    actual = Y[index]
    image = X[:, index].reshape(28, 28)

    print("Prediction:", prediction[0])
    print("Actual:", actual)

    plt.imshow(image, cmap="gray")
    plt.title("Prediction: " + str(prediction[0]) + " | Actual: " + str(actual))
    plt.axis("off")
    plt.show()

while True: #keep allowing user to select images
    index = int(input("Enter an image index (0-999): "))
    make_prediction(index, X_dev, Y_dev)