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

def obtain_empty_confusion_matrix(number_activities:int):
    """
    Creates an empty confusion matrix

    :returns: the new confusion matrix
    :rtype: numpy.array
    """
    return np.zeros((number_activities, number_activities), dtype=np.float64)

def obtain_confusion_matrix(number_activities:int, real_class, obtained_class):
    """
    Obtains an confusion matrix from the given data

    :param number_activities int: number of activities
    :param real_class numpy.array: the real classes
    :param obtained_class numpy.array: the obtained class
    :returns: the new confusion matrix
    :rtype: numpy.array
    """
    #We create the matrix
    confusion_matrix = np.zeros((number_activities, number_activities),
                                dtype=np.float64)
    #We fill in the values
    for real_act, obtained_act in zip(real_class, obtained_class):
        confusion_matrix[real_act, obtained_act] += 1
    #We return the result
    return confusion_matrix
    
def obtain_accuracy(confusion_matrix):
    """
    From a confusion matrix, obtian the accuracy
    
    :param confusion_matrix numpy.array: the confusion matrix
    :returns: the accuracy
    :rtype float
    """
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    
def obtain_precision(confusion_matrix):
    """
    From a confusion matrix, obtain the precision per class
    andthe average
    
    :param confusion_matrix numpy.array: the confusion matrix
    :returns: the precision
    :rtype numpy.array
    """
    pre_activity = np.diag(confusion_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        pre_activity = pre_activity / np.sum(confusion_matrix, axis=1)

    #We may have divisions of 0/0 or n/0, resulting in values nan or inf
    #To fix those, we identify them and replace them by 0
    pre_activity[np.isnan(pre_activity)] = 0.0
    pre_activity[np.isinf(pre_activity)] = 0.0

    #Append the average at the beginning
    return np.concatenate(([np.mean(pre_activity)], pre_activity))

def obtain_recall(confusion_matrix):
    """
    From a confusion matrix, obtain the recall per class
    and the average
    
    :param confusion_matrix numpy.array: the confusion matrix
    :returns: the recall
    :rtype: numpy.array
    """
    rec_activity = np.diag(confusion_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        rec_activity = rec_activity / np.sum(confusion_matrix, axis=0)
    
    #We may have divisions of 0/0 or n/0, resulting in values nan or inf
    #To fix those, we identify them and replace them by 0
    rec_activity[np.isnan(rec_activity)] = 0.0
    rec_activity[np.isinf(rec_activity)] = 0.0
    
    #Append the average at the beginning
    return np.concatenate(([np.mean(rec_activity)], rec_activity))

def obtain_fscore(precision, recall):
    """
    From the obtain precisiona and recall for each class, obtain their fscore
    and the average fscore of all classes
    
    :param precision numpy.array: precision for each class
    :param recall numpy.array: recall for each class
    :returns: the fscore
    :rtype: numpy.array
    """
    fscore_activity = 2 * precision[:] * recall[:]
    with np.errstate(divide='ignore', invalid='ignore'):
        fscore_activity = fscore_activity / (precision[:] + recall[:])
    
    #We may have divisions of 0/0 or n/0, resulting in values nan or inf
    #To fix those, we identify them and replace them by 0
    fscore_activity[np.isnan(fscore_activity)] = 0.0
    fscore_activity[np.isinf(fscore_activity)] = 0.0
    
    #Append the average at the beginning
    return np.concatenate(([np.mean(fscore_activity)], fscore_activity))
    