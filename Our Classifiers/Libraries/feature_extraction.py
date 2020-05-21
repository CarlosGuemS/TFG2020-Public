import datetime, sys
import numpy as np
from math import exp
from itertools import permutations
import random
random.seed(1234)

##Class that represents all possible features
#Same function as tags in C language
class Features:
    SIMPLE_COUNT = 1
    MATRIX_COUNT = 2
    TD_COUNT = 3
    MATRIX_TD_COUNT = 4
    PWA = 5

##Functions to extract the features themselves
def _obtain_seconds_mignight_last_event(window: list):
    """
    Given a window, returns the amount of seconds elapsed between midnight
    and the last event of the window

    :param window list: the window
    :returns: amount of seconds elapsed between midnight and the last event on
    the window
    :rtype: int
    """
    date = window[-1][0]
    return date.second + 60 * date.minute + 360 * date.hour

def _obtain_seconds_mignight_first_event(window: list):
    """
    Given a window, returns the amount of seconds elapsed between midnight
    and the first event of the window

    :param window list: the window
    :returns: amount of seconds elapsed between midnight and the first event on
    the window
    :rtype: int
    """
    date = window[0][0]
    return date.second + 60 * date.minute + 360 * date.hour

def _obtain_week_day_last_event(window: list):
    """
    Given a window, returns the day of the week of the last event of the
    window

    :param window list: the window
    :returns: day of the week (0-6)
    :rtype: int
    """
    date = window[-1][0]
    return date.weekday()

def _obtain_window_seconds_elapsed(window: list):
    """
    Given a window, returns the amount of seconds elapsed between the first
    and last event of the window

    :param window list: the window
    :returns: amount of seconds elapsed between the first and last event of the
    window
    :rtype: int
    """
    first_date = window[0][0]
    last_date = window[-1][0]
    first_time_delta = datetime.timedelta(seconds=first_date.second,
                                          minutes=first_date.minute,
                                          hours=first_date.hour)
    last_time_delta = datetime.timedelta(seconds=last_date.second,
                                          minutes=last_date.minute,
                                          hours=last_date.hour)
    difference = last_time_delta - first_time_delta
    return difference.total_seconds()

def _obtain_simple_count_sensors(window:list, num_sensor:int):
    """
    Given a window, counts the number of times the given sensor appears in it

    :param window list: the window
    :param num_sensor int: the number of sensors to count
    :returns: list of the number of times the given sensor appears in the window
    :rtype: list
    """
    #We create a list, where each position represents a sensor
    sensor_count = [0] * num_sensor
    #We count each sensor appeareance
    for event in window:
        sensor_count[event[1]] += 1
    return sensor_count

def _obtain_time_dependency_count_sensors(window:list, num_sensor:int):
    """
    Given a window, counts the number of times the given sensor appears in it.
    Includes the use of time dependency

    :param window list: the window
    :param num_sensor int: the number of sensors to count
    :returns: the number of times the given sensor appears in the window
    :rtype: list
    """
    #Reference time
    ref_time = window[-1][0]
    #Constant used, exact value obtained from a paper
    td_constant = -2**-3

    #We create a list, where each position represents a sensor
    sensor_count = [0] * num_sensor
    #We count each sensor appeareance
    for event in window:
        sensor_count[event[1]] += exp(td_constant*(ref_time - event[0]).total_seconds())
    return sensor_count

##Class extraction features
def _obtain_class(window:list):
    """
    Given a window, returns the activity of the last event in it
    
    :param window list: the window
    :returns: returns the activity of the last event
    :rtype: int
    """
    return window[-1][-1]

##Obtaining the feature
NUMBER_BASE_FEATURES = 10
def obtain_feature_vector(features:list, window:list, classes:list,
                          num_classes:int, prev_class: int, num_sensor: int,
                          mi):
    """
    Extracts the given features from the given window.
    :raises ValueError: a given feature is not supported

    :param feature list: the feature to be extracted. Must be recognized
    :param window list: the window
    :param classes list: possible classes in the data
    :param num_classes int: number of classes
    :param prev_class int: the class of the window before this one (excluding
    unrecognized activities).
    :param num_sensor int: number of sensors
    :returns: the feature vector
    :rtype: list
    """

    #Add base features
    #Day of the week is implemented using one-hot enconding
    day_week = _obtain_week_day_last_event(window)
    feature_vector = [ int(i==day_week) for i in range(1,8) ]
    feature_vector.append(_obtain_seconds_mignight_last_event(window))
    feature_vector.append(_obtain_seconds_mignight_first_event(window))
    feature_vector.append(_obtain_window_seconds_elapsed(window))

    #Add additional features
    for feature in features:
        #Simple Count
        if feature == Features.SIMPLE_COUNT:
            #We have to explore every posible sensor
            feature_vector += _obtain_simple_count_sensors(window, num_sensor)
        
        #Previous Window Activity 
        elif feature == Features.PWA:
            #Previous Window Activity
            # If now previous activity is detected, make it so the previous
            # class is 'chosen' at random from the rest 
            pwa = [int(x_class == prev_class) for x_class in classes]
            if not any(pwa):
                pwa = [1/num_classes for x_class in classes]
            feature_vector = feature_vector + pwa
        
        #Mutual Information matrix Count
        elif feature == Features.MATRIX_COUNT:
            if mi is None:
                raise ValueError("No Mutual Information matrix is defined!")
            #We have to explore every posible sensor
            count = _obtain_simple_count_sensors(window, num_sensor)
            #We multiply the count by the coefficient in the MI matrix
            mi_count = count * mi[:, window[-1][1]]
            #We append the results 
            feature_vector += list(mi_count) 
        
        #Mutual Information matrix + Time dependency count
        elif feature == Features.MATRIX_TD_COUNT:
            if mi is None:
                raise ValueError("No Mutual Information matrix is defined!")
            #We have to explore every posible sensor
            count = _obtain_time_dependency_count_sensors(window, num_sensor)
            #We multiply the count by the coefficient in the MI matrix
            mi_count = count * mi[:, window[-1][1]]
            #We append the results 
            feature_vector += list(mi_count)
        
        #Time dependency count
        elif feature == Features.TD_COUNT:
            #We have to explore every posible sensor
            feature_vector += _obtain_time_dependency_count_sensors(window,
                                                                    num_sensor)
            
        else:
            #Unrecognized sensor
            raise ValueError("Unrecognized feature: \"" + feature + "\"")
    return feature_vector

def obtain_event_segmentation_data(data:list, features:str, num_sensor:int,
                                   num_classes:int, mi = None):
    """
    Given data and window size, obtain the features through the given window
    size

    :param data list: the data, already segmented into windows
    :param window_size int: window size
    :param num_sensor int: number of sensors in the data
    :param num_classes int: number of possible classes in the data
    :param mi numpy.array: mutual information matrix (if applicable)
    :returns: array of features and the corresponding array of classes
    :rtype: numpy.array, numpy.array
    """
    #Obtain the number of features:
    num_features = NUMBER_BASE_FEATURES
    num_features += num_classes*(Features.PWA in features)
    num_features += num_sensor*(Features.SIMPLE_COUNT in features)
    num_features += num_sensor*(Features.MATRIX_COUNT in features)
    num_features += num_sensor*(Features.MATRIX_TD_COUNT in features)
    num_features += num_sensor*(Features.TD_COUNT in features)

    #We prepare where to store the data
    temp_data = np.zeros((len(data), num_features))
    temp_class = np.zeros((len(data)), dtype=int)
    classes = np.linspace(0, num_classes-1, num_classes)
    last_class = num_classes;
    
    #We cover all the possible windows
    for window_index, window in enumerate(data):
        temp_data[window_index] = obtain_feature_vector(features, window,
                                                        classes, num_classes,
                                                        last_class,
                                                        num_sensor, mi)
        temp_class[window_index] = _obtain_class(window)
        last_class = temp_class[window_index] if temp_class[window_index] != num_classes else last_class
    return temp_data, temp_class

def segment_data(data:list, window_size:int):
    """
    Given the data it returns the data segmented into windows

    :param data list: the data. The data should come wrapped in a list, as to
    partition the data in a way they don't create windows between each other
    :param window_size int: window size
    :returns: data in windows
    :rtype: list
    """
    windows = []
    for partition in data:
        windows += [partition[ii:ii+window_size]
                    for ii in range(len(partition)-window_size+1)]
    return windows


##Sensor Windows Mutual Information Extension
def obtain_mutual_information_ext_matrix(windows: list, number_of_sensors:int):
    """
    Builds a mutual information adjacency extended matrix using the given data.

    :param windows list: the data from where to build the adjacency matrix.
    The data must be sorted into windows already
    :param number_of_sensors int: number of possible sensors. Also the number
    of rows and columns
    :returns: the mutual information adjacency matrix
    :rtype: numpy.array
    """

    mi = np.zeros((number_of_sensors, number_of_sensors))

    #We analize each window
    for window in windows:
        #We obtain all the possible combinations of sensors within the window
        sensors_in_window = set(map(lambda event: event[1], window))
        #We analyze all possible permutations
        for sensor1, sensor2 in permutations(sensors_in_window, 2): 
            mi[sensor1, sensor2] += 1
        #We must also consider sensor activating with themselves
        for sensor in sensors_in_window:
            mi[sensor, sensor] += 1
    
    #Normalization
    mi = mi / len(windows)
    return mi

##Limit of training samples
def limit_training_samples(samples, samples_classes, num_classes:int,
                           max_samples_total:int, max_samples_class:int):
    """
    Returns a set of sample and classes limited to given number of samples.
    If there are more samples than can be used they are chosen randomply.

    :param samples numpy.array: the samples used
    :param samples_classes: the classes of the samples. They must correspond
    to the samples in 'samples'
    :param num_classes int: number of possible classes in the data
    :param max_samples_total int: maximum number of samples
    :param max_samples_class int: maximum number of samples per class
    """

     #We check we don't have too many samples:
    if samples_classes.size > max_samples_total:
        #We need to limit the amount of samples for SVM to 1000
        limited_training_data = np.empty((0,samples.shape[1]))
        limited_training_class = np.empty(0)
        for c in range(num_classes):
            #Indices of class c
            class_indices = np.where(samples_classes == c)[0]
            #We check if there are more than MAX_SAMPLES_PER_CLASS
            if class_indices.size > max_samples_class:
                #If there are more than 100 we must limit the number
                # of samples to 1000
                class_indices = random.sample(list(class_indices),
                                              max_samples_class)
            #We concatenate the selected samples to the activity set
            limited_training_data = np.concatenate((limited_training_data,
                                                    samples[class_indices]),
                                                   0)
            limited_training_class = np.concatenate((limited_training_class,
                                                     samples_classes[class_indices]),
                                                    0)
        #We return the results
        return limited_training_data, limited_training_class
    else:
        #We don't exceed the maxium
        return samples, samples_classes