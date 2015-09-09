import numpy as np
import math


def sigmoid(z):
    return 1. / (1. + math.exp(-z))


sigmoid_vector = np.vectorize(sigmoid)


def cost_function(x, y, theta, lmbda):
    """
    J(theta) = (-1 / m) * sum_by_i_from_1_to_m(y(i) * log(h(x(i))) + (1 - y(i)) * log(1 - h(x(i)))) + lambda / (2 * m) * sum_by_j_from_1_to_n(theta(j))
    where:
    m - count of training examples
    x(i) - i-th training example
    h(x(i)) - output for i-th training example produced by our algorithm
    y(i) - expected output for i-th training example
    lambda - regularization parameter
    n - features count
    """
    m = x.shape[0]
    return (-1. / m) * (y.T * np.log(sigmoid_vector(x * theta.T)) + (1. - y) * np.log(1. - sigmoid_vector(x * theta.T))) + lmbda / (2. * m) * (theta[2:] * theta[2:].T)


def get_gradient(x, y, theta, lmbda):
    """
    G(0) = (-1 / m) * sum_by_i_from_1_to_m((h(x(i)) - y(i)) * x(i,j))
    G(j) = (-1 / m) * sum_by_i_from_1_to_m((h(x(i)) - y(i)) * x(i,j)) + (lambda / m) * theta(j) for j = 1..n
    where:
    m - count of training examples
    lambda - regularization parameter
    """
    m = x.shape[0]
    n = x.shape[1]
    gradient = (-1. / m) * ((sigmoid_vector(x * theta.T) - y).T * x) + (lmbda / m) * theta
    gradient[0] -= (lmbda / m) * theta[0]
    return gradient
