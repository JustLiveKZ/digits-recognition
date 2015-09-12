import numpy as np
import math


@np.vectorize
def sigmoid(z):
    try:
        return 1. / (1. + math.exp(-z))
    except OverflowError:
        # OverflowError is raised in z is very big negative value
        # so math.exp(-z) evaluates as very big positive value
        # In this case return very small positive value that close to 0.0
        return 1e-9


@np.vectorize
def log(z):
    try:
        return math.log(z)
    except ValueError:
        # ValueError is raised if z is 0.0
        # In this case return very big negative value
        return -1e9


def compute_cost(theta, x, y, lmbda):
    """
    J(theta) = (-1 / m) * sum_for_i=1_to_m(y[i] * log(h(x[i])) + (1 - y[i]) * log(1 - h(x[i]))) +
    + lambda / (2 * m) * som_for_j=1_to_n(theta[j] ** 2)

    where:
    m - count of training examples
    x[i] - i-th training example
    h(x[i]) - output for i-th training example produced by algorithm
    y[i] - expected output for i-th training example
    lambda - regularization parameter
    n - features count (excluding artificially added ones as first feature)
    """
    m = x.shape[0]
    theta = theta.reshape((theta.shape[0], 1))
    J = (-1. / m) * (np.dot(y.T, log(sigmoid(np.dot(x, theta))))
                     + np.dot((1. - y).T, log(1. - sigmoid(np.dot(x, theta))))) \
        + lmbda / (2. * m) * np.dot(theta[1:].T, theta[1:])
    return J[0, 0]


def compute_gradient(theta, x, y, lmbda):
    """
    gradient[0] = (1 / m) * sum_for_i=1_to_m((h(x[i]) - y[i]) * x[i,j])
    gradient[j] = (1 / m) * sum_for_i=1_to_m((h(x[i]) - y[i]) * x[i,j]) +
    + (lambda / m) * theta(j) for j = 1..n
    where:

    m - count of training examples
    x[i] - i-th training example
    h(x[i]) - output for i-th training example produced by algorithm
    y[i] - expected output for i-th training example
    lambda - regularization parameter
    x[i,j] - j-th feature of i-th training example
    n - features count (excluding artificially added ones as first feature)
    """
    m = x.shape[0]
    n = x.shape[1]
    theta = theta.reshape((theta.shape[0], 1))
    gradient = (1. / m) * (np.dot((sigmoid(np.dot(x, theta)) - y).T, x)).T + (lmbda / m) * theta
    gradient.shape = n
    gradient[0] -= (lmbda / m) * theta[0, 0]
    return gradient


def predict(all_theta, x):
    """
    Predicts output for training set x
    """
    predictions = np.argmax(sigmoid(np.dot(x, all_theta.T)), axis=1)
    predictions.shape = (predictions.shape[0], 1)
    return predictions
