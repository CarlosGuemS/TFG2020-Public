import numpy as np


##Discretization of data (before feature extraction)

def set_all_to_1(data:list):
    """
    Given the data, sets all their messages to 1

    :param data list: list of data
    :rtype: None
    """
    for event in data:
        if int(event[2]) != event[2]:
            event[2] = 1.0

##Discretization of features (after feature extraction)

def obtain_categorization_function(training_data, num_cat:list):
    """
    Returns a function that discretizes data into equiprobable categories
    based on the training data used.

    :param training_data numpy.array: data to be discretized
    :param num_cat: number of categories to be used
    :returns: function which performs the discretizatoin
    :rtype: function
    """
    thresholds = np.empty_like([],shape=(len(training_data[0]), num_cat-1))

    #Fiven the training data, we must now find the thresholds
    for category in range(0, len(training_data[0])):
        #We select the data from a feature and sort it
        category_data = training_data[:,category]
        category_data.sort()

        #We divide the sorted data into the given number of categories
        categories_data_split = np.array_split(category_data, num_cat)

        #We build the thresholds, getting the biggest value of each category
        #except the last
        thresholds[category, :] = np.array([max(cat) for cat in categories_data_split[:-1]])
    
    #With the thresholds built, we can now define the funciton to return
    def discretization(data):
        """
        Given data in the same format as the one used in
        obtain_categorization_function, return the data after being
        discretized

        :param data numpy.array: data to be discretized
        
        """
        for feature in data:
            for ii in range(len(feature)):
                jj = 0
                while (jj < len(thresholds[ii])
                      and feature[ii] > thresholds[ii][jj]):
                      jj += 1
                feature[ii] = jj+1

    return discretization              