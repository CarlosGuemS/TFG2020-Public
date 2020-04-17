import datetime, sys
import numpy as np
from math import exp

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
NUMBER_BASE_FEATURES = 4
def obtain_feature_vector(features:list, window:list, prev_class: int,
                          num_sensor: int, mi):
    """
    Extracts the given features from the given window.
    :raises ValueError: a given feature is not supported

    :param feature list: the feature to be extracted. Must be recognized
    :param window list: the window
    :param prev_class int: the class of the window before this one (excluding
    unrecognized activities).
    :param num_sensor int: number of sensors
    :returns: the feature vector
    :rtype: list
    """

    #Add base features
    feature_vector = []
    feature_vector.append(_obtain_week_day_last_event(window))
    feature_vector.append(_obtain_seconds_mignight_last_event(window))
    feature_vector.append(_obtain_seconds_mignight_first_event(window))
    feature_vector.append(_obtain_window_seconds_elapsed(window))

    #Add additional features
    for feature in features:
        if feature == "SIMPLE COUNT SENSOR":
            #We have to explore every posible sensor
            for sensor in range(num_sensor):
                feature_vector.append(_obtain_simple_count_sensors(window,
                                                                   sensor))
        elif feature == "PREV CLASS LAST EVENT":
            #If no window exists, this feature is labeled as other (-1)
            temp = prev_class
            feature_vector.append(temp)

        elif feature == "MATRIX COUNT SENSOR":
            if mi is None:
                raise ValueError("No Mutual Information matrix is defined!")
            #We have to explore every posible sensor
            for sensor in range(num_sensor):
                count = _obtain_simple_count_sensors(window, sensor)
                #We multiply the count by the coefficient in the MI matrix
                feature_vector.append(count * mi[sensor, window[-1][1]])

        elif feature == "TIME DEPEDENCY MATRIX COUNT SENSOR":
            if mi is None:
                raise ValueError("No Mutual Information matrix is defined!")
            #We have to explore every posible sensor
            for sensor in range(num_sensor):
                count = _obtain_time_dependency_count_sensors(window, sensor)
                #We multiply the count by the coefficient in the MI matrix
                feature_vector.append(count * mi[sensor, window[-1][1]])

        elif feature == "TIME DEPEDENCY COUNT SENSOR":
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
    :param num_classes int: number of classes in the data
    :param mi numpy.array: mutual information matrix (if applicable)
    :returns: array of features and the corresponding array of classes
    :rtype: numpy.array, numpy.array
    """
    temp_data = []; temp_class = []
    last_class = num_classes
    #We cover all the possible windows
    for window in data:
        temp_data.append(obtain_feature_vector(features, window, last_class,
                                               num_sensor, mi))
        temp_class.append(_obtain_class(window))
        last_class = temp_class[-1] if temp_class[-1] else last_class
    return np.array(temp_data), np.array(temp_class)

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