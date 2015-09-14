# Handwritten digits recognition

This is an application that recognizes handwritten digits.
Application uses [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/).

Algorithm used in application is multi-class logistic regression.

### Usage
#### Training
`python main.py train`

This command is used to train classifiers. It uses optimization parameters defined in `parameters.py` (such as optimization function, max iterations, regularization parameter). You can play around with these values to try to get better results.

##### Repository contains some examples of trained classifiers
`theta_cg.txt` - Produced by "Conjugate gradient" optimization method with 100 iterations

`theta_bfgs.txt` - Produced by "BFGS" optimization method with 100 iterations

`theta_ncg.txt` - Produced by "Newton-CG" optimization method with 100 iterations

#### Testing
`python main.py test`

This command is used to test how successful algorithm can recognize handwritten digits.

##### Results for example trained classifiers described above
`theta_cg.txt` - 91.93% (9193 correctly predicted digits out of 10000)

`theta_bfgs.txt` - 91.12% (9112 out of 10000)

`theta_ncg.txt` - 84.82% (8482 out of 10000)
