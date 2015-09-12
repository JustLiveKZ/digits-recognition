import numpy as np

from parameters import classifiers_number, features_number, lmbda, training_examples_count, theta_filename, optimization_function, max_iterations
from functions import compute_cost, compute_gradient


def train(images, labels):
    try:
        print 'Trying to read parameters from "%s"' % theta_filename
        np.fromfile(theta_filename, sep='\n').reshape((classifiers_number, features_number + 1))
        print 'There is already exists "%s" that contains trained classifiers' % theta_filename
        print 'If you want to re-train classifiers rename or delete "%s"' % theta_filename
    except (IOError, ValueError) as e:
        print 'File not found or corrupted. Training will recover it'
        # Adding fake feature
        images = np.concatenate((np.ones((images.shape[0], 1)), images), axis=1)
        all_theta = np.zeros((classifiers_number, features_number + 1))
        for i in range(classifiers_number):
            print "Training classifier for digit %d" % i
            mapped_labels = np.array(map(lambda label: int(label == i), labels)) \
                .reshape((training_examples_count, 1))
            # Pass cost function as positional parameter
            # because in some minimization functions
            # it is called "f" and in others "func"
            theta = optimization_function(f=compute_cost,
                                          x0=all_theta[i],
                                          args=(images, mapped_labels, lmbda),
                                          fprime=compute_gradient,
                                          maxiter=max_iterations)
            all_theta[i] = theta
        print "Training was completed. Recording trained classifiers"
        with open(theta_filename, 'w') as f:
            for item in all_theta.reshape(all_theta.shape[0] * all_theta.shape[1]):
                f.write('%s\n' % item)
        print "Trained classifiers were recorded into '%s'" % theta_filename
