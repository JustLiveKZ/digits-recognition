import scipy.optimize

training_examples_count = 60000
testing_examples_count = 10000
image_width = 28
image_height = 28
features_number = image_width * image_height
classifiers_number = 10
theta_filename = 'theta.txt'

"""
Regularization parameter
Do not set it to big value cause it leads to underfitting
"""
lmbda = 0.1

"""
Optimization method that will be used to find
the minimum value of cost function

You can try several methods to determine which
one fits better than others

Some of available minimization functions:
scipy.optimize.fmin_cg - Conjugate gradient method
scipy.optimize.fmin_bfgs - Broyden, Fletcher, Goldfarb, and Shanno (BFGS) method
scipy.optimize.fmin_ncg - Newton-CG method

More minimization functions you can find here
http://docs.scipy.org/doc/scipy/reference/optimize.html
"""
optimization_function = scipy.optimize.fmin_bfgs
max_iterations = 100
