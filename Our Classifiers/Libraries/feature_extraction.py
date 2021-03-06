import datetime, sys
import numpy as np
from math import exp
from itertools import permutations, chain
import random
random.seed(1234)

##Class that represents all possible features
#Same function as tags in C language
class Features:
    SIMPLE_COUNT = 1
    MATRIX_COUNT = 2
    TD_COUNT = 3
    MATRIX_TD_COUNT = 4

##Functions to extract the features themselves
## Non-sequential models
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

##Obtaining the feature
NUMBER_BASE_FEATURES = 10
def obtain_event_segmentation_data(data:list, feature:int, num_sensor:int):
    """
    Given data and window size, obtain the features through the given window
    size

    :param data list: the data, already segmented into windows
    :param feature int: the feature code
    :param num_sensor int: number of sensors in the data
    :returns: array of features, the corresponding array of classes
    and the last event of each window
    :rtype: numpy.array, numpy.array, numpy.array
    """
    #Obtain the number of features:
    num_features = NUMBER_BASE_FEATURES + num_sensor

    #We prepare where to store the data
    temp_data = np.zeros((len(data), num_features))
    temp_class = np.zeros((len(data)), dtype=int)
    last_windows = np.zeros((len(data)), dtype=int)
    
    #We cover all the possible windows
    for window_index, window in enumerate(data):
        #Obtain the class
        temp_class[window_index] = window[-1][-1]

        #We obtain the last event
        last_windows[window_index] = window[-1][1]

        ##Obtain the feature vector
        day_week = _obtain_week_day_last_event(window)
        temp_data[window_index, 0:7] = [ int(i==day_week) for i in range(1,8) ]
        temp_data[window_index, 7] = _obtain_seconds_mignight_last_event(window)
        temp_data[window_index, 8] = _obtain_seconds_mignight_first_event(window)
        temp_data[window_index, 9] = _obtain_window_seconds_elapsed(window)

        #Add the correct sensor count depending on the additional features
        #Simple Count or Mutual Information matrix Count
        if feature == Features.SIMPLE_COUNT or feature == Features.MATRIX_COUNT:
            #We have to explore every posible sensor
            temp = _obtain_simple_count_sensors(window, num_sensor)
            temp_data[window_index, NUMBER_BASE_FEATURES : num_features] = temp
        
        #Time dependency count
        # or Mutual Information matrix + Time dependency count
        elif feature == Features.TD_COUNT or feature == Features.MATRIX_TD_COUNT:
            #We have to explore every posible sensor
            temp = _obtain_time_dependency_count_sensors(window, num_sensor)
            temp_data[window_index, NUMBER_BASE_FEATURES : num_features] = temp 
            
        else:
            #Unrecognized sensor
            raise ValueError("Unrecognized feature: \"" + feature + "\"")
    
    ##End window loop
    return temp_data, temp_class, last_windows

def apply_sensor_sensor_dependency(data, last_event, mi, num_sensor:int):
    """
    Given already segmented data, apply the MI to it

    :param data numpy.array:
    :param last_event list: the list of last events
    :param mi numpy.array: mutual information matrix (if applicable)
    :param num_sensor int: number of sensors in the data
    :returns: a copy of the data with the mi applied
    :rtype: numpy.array
    """
    num_features = NUMBER_BASE_FEATURES + num_sensor

    #We first create a copy of the data
    #and copy the features unaffected
    data_copy = np.empty(data.shape)
    data_copy[:, :NUMBER_BASE_FEATURES] = data[:, :NUMBER_BASE_FEATURES]
    #We now modify each count
    for feature_index in range(data.shape[0]):
        #We obtain the sensor count
        temp = data[feature_index, NUMBER_BASE_FEATURES : num_features]
        #We apply the MI
        temp = temp * mi[:, last_event[feature_index]]
        #We store it once again
        data_copy[feature_index, NUMBER_BASE_FEATURES : num_features] = temp
    
    return data_copy

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
def obtain_mutual_information_ext_matrix(windows: list, windows_indexes, number_of_sensors:int):
    """
    Builds a mutual information adjacency extended matrix using the given data.

    :param windows list: the data from where to build the adjacency matrix.
    The data must be sorted into windows already
    :param windows_indexes numpy.array: the of indices indicating the training
    windows
    :param number_of_sensors int: number of possible sensors. Also the number
    of rows and columns
    :returns: the mutual information adjacency matrix
    :rtype: numpy.array
    """

    mi = np.zeros((number_of_sensors, number_of_sensors))

    #We analize each window
    for win_index in windows_indexes:
        #Obtain the window
        window = windows[win_index]
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
                           max_samples_total:int):
    """
    Returns a set of sample and classes limited to given number of samples.
    If there are more samples than can be used they are chosen randomply.

    :param samples numpy.array: the samples used
    :param samples_classes: the classes of the samples. They must correspond
    to the samples in 'samples'
    :param num_classes int: number of possible classes in the data
    :param max_samples_total int: maximum number of samples
    :returns: the limit set of samples
    :rtype: numpy.array, numpy.array 
    """

     #We check we don't have too many samples:
    if samples_classes.size > max_samples_total:
        #We need to limit the amount of samples for SVM to 1000
        limited_training_data = np.empty((0,samples.shape[1]))
        limited_training_class = np.empty(0)
        max_samples_class = max_samples_total // num_classes
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


## For sequential models
def extract_sensor_chains(event_chains: list):
    """
    From a list of list of events, obtain the chains of sensors and their classes
    :param event_chains list: chains of events
    :returns: a generator with the chains of sensors and their classes
    :rtype: generator
    """
    for event_chain in event_chains:
        #We extact the sensora
        sensor_chain = [['sensor='+str(event[1])] for event in event_chain]
        #Divide sensors and classes
        class_chain = [str(event[-1]) for event in event_chain]
        #Yield the results
        yield sensor_chain, class_chain
