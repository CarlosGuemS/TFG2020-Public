import sys

import numpy as np

from general import msg_error

def obtain_classifier_prediction(num_classes:int, test_data,
                                 predict_function, scaler = None,
                                 normalize_function = None):
    """
    Returns the accuracy score of a scikit-learn classifier (originally
    designed for naive bayes) for the given samples.

    :param num_classes list: number of classes of the classifier
    :param test_data numpy.array: test partition samples
    :param test_class numpy.array: test_data's classes
    :param predict_function function: function that makes the prediction by
    the trained classifer
    :param scaler sklearn.preprocessing.Scaler: trained scaler
    :param normalize_function function: function that normalizes the data
    :returns: the accuracy score and the predicted activities
    :rtype: float, numpy.array
    """

    if scaler:
        test_data = scaler.transform(test_data)
    if normalize_function:
        normalize_function(test_data)
    return np.array(predict_function(test_data), dtype=np.int8)

def obtain_prior_probabilitites(sample_classes, num_activities:int):
    """
    Obtain the prior probabilitites of the training samples.
    The classes must be normalized (range from 0 to C-1, where C is the number
    of classes)

    :param sample_classes numpy.array: training classes
    :param num_activitites int: number of activities
    """
    prior_probs = np.zeros(shape=(num_activities))
    for i in sample_classes:
        prior_probs[i] = prior_probs[i] + 1
    return prior_probs / sample_classes.size