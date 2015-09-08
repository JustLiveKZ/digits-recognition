import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def read(dataset="training", path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the numpy.uint8 2D array of pixel
    data for the given image and the second element being a label.
    """
    if dataset is "training":
        images_filename = os.path.join(path, 'train-images-idx3-ubyte')
        labels_filename = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        images_filename = os.path.join(path, 't10k-images-idx3-ubyte')
        labels_filename = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(labels_filename, 'rb') as f:
        magic, count = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    with open(images_filename, 'rb') as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), rows, cols)

    # Create an iterator which returns each image in turn
    for i in range(len(labels)):
        yield (images[i], labels[i])


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
