import numpy as np
from itertools import permutations

#Mutual information matrix
def obtain_mutual_information_matrix(data:list, number_of_sensors:int):
    """
    Builds a mutual information adjacency matrix using the given data.

    :param data list: the data from where to build the adjacency matrix
    :param number_of_sensors int: number of possible sensors. Also the number
    of rows and columns
    :returns: the mutual information adjacency matrix
    :rtype: numpy.array
    """
    
    mi = np.zeros((number_of_sensors, number_of_sensors))
    
    #Counting the adjency:
    for i in range(len(data)-1):
        mi[data[i][1]][data[i+1][1]] = 1
    
    #Normalizing
    mi = mi / len(data);

    return mi

#Sensor Windows Mutual Information Extension
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