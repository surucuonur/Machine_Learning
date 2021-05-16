import numpy as np
import matplotlib.pyplot as plt


class Gradient_Descent:
    def __init__(self, x, target, learning_rate):
        self.x = x
        self.target = target
        self.learning_rate = learning_rate
        self.w = np.random.rand()

    def forward_pass(self):
        # Forward Pass
        self.sop = self.w * self.x  # Dot product (SOP)
        self.predicted = 1.0 / (
            1 + np.exp(-1 * self.sop)
        )  # Activasion of neuron with sigmoid
        self.err = np.power(
            self.predicted - self.target, 2
        )  # Error between predicted value vs target

    def backward_pass(self):
        # Backward Pass
        g1 = 2 * (self.predicted - self.target)  # (d)error / (d)predicted
        g2 = self.predicted * (1.0 - self.predicted)  # (d)predicted / (d) sop
        g3 = self.x  # (d) sop / (d) W
        self.grad = g3 * g2 * g1  # (d) error / (d) W
        self.w = self.w - self.learning_rate * self.grad

    def update_weight(self):
        self.w = self.w - self.learning_rate * self.grad


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Creating a random weight with specified matrix size
        # n_inputs = number of inputs
        # n_neurons = number of neurons
        # randn: gaussian distribution rounded around 0
        # We dont need to transpose the weight when doing dot product
        self.weights = 0.10 * np.random.randn(
            n_inputs, n_neurons
        )  # Multiplying by 0*10 to achieve values less than 1

        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Activation of each neurons in the hidden layer
class Activation_ReLu:
    def forward(self, inputs=None):
        if inputs is None:
            inputs = []
        self.output = np.maximum(0, inputs)


# Input, Output, Learning rate are being fed
G_D = Gradient_Descent(0.1, 0.3, 0.5)
layer1 = Layer_Dense(1, 5)
layer1.forward(0.1)


y_axis = []
for k in range(80000):
    G_D.forward_pass()
    G_D.backward_pass()
    G_D.update_weight()
    y_axis.append(G_D.err)


x_axis = np.arange(k + 1)  # Linspace 1:k+1 initialized as an x axis
plt.plot(x_axis, y_axis)  # Plots the error vs number of steps
plt.show()
