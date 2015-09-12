import numpy as np

from functions import predict
from parameters import classifiers_number, features_number, theta_filename


def test(images, labels):
    try:
        all_theta = np.fromfile(theta_filename, sep='\n').reshape((classifiers_number, features_number + 1))
        # Adding fake feature
        images = np.concatenate((np.ones((images.shape[0], 1)), images), axis=1)
        predictions = predict(all_theta, images)
        correct_predictions = sum(map(int, (labels == predictions).reshape(labels.shape[0])))
        print '*** Testing results ***'
        print 'Total digits: %d' % images.shape[0]
        print 'Correctly predicted: %d' % correct_predictions
        print 'Percentage: %.2f%%' % (correct_predictions * 100. / images.shape[0])
    except (IOError, ValueError) as e:
        print 'Failed to read parameters from "%s". Error: %s' % (theta_filename, str(e))
        print 'File does not exists or corrupted. Run "train" command to recover it'
