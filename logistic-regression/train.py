import numpy as np
from constants import CLASSIFIERS_NUMBER, FEATURES_NUMBER
import mnist


def train():
    images, labels = mnist.read('training', '..')
    initial_theta = np.zeros((CLASSIFIERS_NUMBER, FEATURES_NUMBER + 1))
    

if __name__ == '__main__':
    train()
