import datetime, sys
import numpy as np
from math import exp
from itertools import permutations

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

def _obtain_simple_count_sensors(window:list, sensor:int):
    """
    Given a window, counts the number of times the given sensor appears in it

    :param window list: the window
    :param sensor int: the sensor to count
    :returns: the number of times the given sensor appears in the window
    :rtype: int
    """
    return sum(1 for event in window if event[1]==sensor)

def _obtain_time_dependency_count_sensors(window:list, sensor:int):
    """
    Given a window, counts the number of times the given sensor appears in it.
    Includes the use of time dependency

    :param window list: the window
    :param sensor int: the sensor to count
    :returns: the number of times the given sensor appears in the window
    :rtype: int
    """
    #Reference time
    ref_time = window[-1][0]
    #Constant used, exact value obtained from a paper
    td_constant = -2**-3

    return sum(exp(td_constant* (ref_time - event[0]).total_seconds()) 
               for event in window if event[1]==sensor)

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
            for sensor in range(num_sensor):
                feature_vector.append(_obtain_simple_count_sensors(window,
                                                                   sensor))
        
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
            for sensor in range(num_sensor):
                count = _obtain_simple_count_sensors(window, sensor)
                #We multiply the count by the coefficient in the MI matrix
                feature_vector.append(count * mi[sensor, window[-1][1]])
        
        #Mutual Information matrix + Time dependency count
        elif feature == Features.MATRIX_TD_COUNT:
            if mi is None:
                raise ValueError("No Mutual Information matrix is defined!")
            #We have to explore every posible sensor
            for sensor in range(num_sensor):
                count = _obtain_time_dependency_count_sensors(window, sensor)
                #We multiply the count by the coefficient in the MI matrix
                feature_vector.append(count * mi[sensor, window[-1][1]])
        
        #Time dependency count
        elif feature == Features.TD_COUNT:
            #We have to explore every posible sensor
            for sensor in range(num_sensor):
                temp = _obtain_time_dependency_count_sensors(window, sensor)
                feature_vector.append(temp)
            
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