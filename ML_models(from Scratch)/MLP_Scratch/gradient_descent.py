# https://blog.paperspace.com/part-1-generic-python-implementation-of-gradient-descent-for-nn-optimization/

import numpy
import matplotlib.pyplot as plt


def sigmoid(sop):
    return 1.0 / (1 + numpy.exp(-1 * sop))


def error(predicted, target):
    return numpy.power(predicted - target, 2)


def error_predicted_deriv(predicted, target):
    return 2 * (predicted - target)


def activation_sop_deriv(sop):
    return sigmoid(sop) * (1.0 - sigmoid(sop))


def sop_w_deriv(x):
    return x


def update_w(w, grad, learning_rate):
    return w - learning_rate * grad


# Initializing the parameters
x = 0.1  # The input
target = 0.3  # The output (target)
learning_rate = 0.5  # Learning Rate
w = numpy.random.rand()  # Weight is randomly being picked
print("Initial W : ", w)

# Forward Pass
y = w * x  # Dot product (SOP)
predicted = sigmoid(y)  # Activasion of neuron with sigmoid
err = error(predicted, target)  # Error between predicted value vs target

# Backward Pass
g1 = error_predicted_deriv(predicted, target)  # (d)error / (d)predicted
g2 = activation_sop_deriv(predicted)  # (d)predicted / (d) sop
g3 = sop_w_deriv(x)  # (d) sop / (d) W
grad = g3 * g2 * g1  # (d) error / (d) W
print(predicted)  # the SOP value

w = update_w(w, grad, learning_rate)  # Updating the weight

y_axis = []
for k in range(1):
    # Forward Pass
    y = w * x
    predicted = sigmoid(y)
    err = error(predicted, target)

    # Backward Pass
    g1 = error_predicted_deriv(predicted, target)
    g2 = activation_sop_deriv(predicted)
    g3 = sop_w_deriv(x)
    grad = g3 * g2 * g1
    print(predicted)

    w = update_w(w, grad, learning_rate)
    y_axis.append(err)

print(y_axis)


x_axis = numpy.arange(k + 1)
plt.plot(x_axis, y_axis)
plt.show()

