import sys, re
import numpy as np
from os import path

#Scikit learn
from sklearn.naive_bayes import GaussianNB

#Preprocessing
sys.path.append('../Libraries')
import feature_extraction as fe
import mutual_information as mi

#Output
sys.path.append('../Evaluation')
from evaluation import Evaluation, confusion_matrix

#Datasets
sys.path.append('../Datasets')
import Kyoto1, Kyoto2, Kyoto3, Aruba

#Constants
WINDOW_SIZES = [5, 12, 19]

##STR that indicates usage
USAGE = "Usage: naive_bayes <feature_mode> <output_name> <data> [--posterior]"

##Auxiliary methods
def _msg_error(msg:str):
    """
    Prints an error in the stderr output and ends the execution prematurely

    :param msg str: error message
    :rtype: None
    """
    print(msg, file=sys.stderr)
    try:
        output.close()
        confusion_file.close()
    except:
        pass
    sys.exit(1)

def flatten(l:list):
    """
    Given a list made of lists, return the concatenation of its inner lists

    :param l list: the list to flatten
    :returns: the flattened list
    :rtype: list
    """
    result = []
    for x in l:
        result += x.copy()
    return result 

def _string__list_to_int_gen(l:list):
    """
    Given converts a list of string to a generator of int
    
    :param l list: list of string
    :returns: generator of int
    :rtype: map object
    """
    return map(lambda x: int(x), l)

def _string__list_to_float_gen(l:list):
    """
    Given converts a list of string to a generator of float
    
    :param l list: list of string
    :returns: generator of float
    :rtype: map object
    """
    return map(lambda  x: float(x), l)

"""
Format of the input configuration: additional features. Possible options:
-BASE: no changes; simple count
-PREV_CLASS: adds the class of the previous window
-TIME_DEPENDENCY: adds time dependency in the count
-MI: mutual information matrix (sensor event adjency)
-MI_EXT: mutual information extened matrix (sensor event window adjency)
"""

#Main
if __name__ == "__main__":
    #Reading the inputs
    if len(sys.argv) < 4:
        _msg_error(USAGE)
    
    #Configuring the features
    if sys.argv[1] == "BASE":
        feature_vector = ["SIMPLE COUNT SENSOR"]
    elif sys.argv[1] == "PWC":
        feature_vector = ["SIMPLE COUNT SENSOR"]
        feature_vector = ["PREV CLASS LAST EVENT"]
    elif sys.argv[1] == "TD":
        feature_vector = ["TIME DEPEDENCY COUNT SENSOR"]
    elif sys.argv[1] == "MI" or sys.argv[1] == "EMI":
        feature_vector = ["MATRIX COUNT SENSOR"]
    elif sys.argv[1] == "PWC+TD":
        feature_vector = ["TIME DEPEDENCY COUNT SENSOR",
                          "PREV CLASS LAST EVENT"]
    elif sys.argv[1] == "PWC+EMI":
        feature_vector = ["MATRIX COUNT SENSOR", "PREV CLASS LAST EVENT"]
    elif sys.argv[1] == "TD+EMI":
        feature_vector = ["TIME DEPEDENCY MATRIX COUNT SENSOR"]
    elif sys.argv[1] == "PWC+TD+EMI":
        feature_vector = ["TIME DEPEDENCY MATRIX COUNT SENSOR",
                          "PREV CLASS LAST EVENT"]
    else:
        _msg_error("Unkown feature: " + sys.argv[1])

    #Selecting the dataset
    dataset = sys.argv[3]
    if dataset == "KYOTO1":
        dataset = Kyoto1
    elif dataset == "KYOTO2":
        dataset = Kyoto2
    elif dataset == "KYOTO3":
        dataset = Kyoto3
    elif dataset == "ARUBA":
        dataset = Aruba
    else:
        #Unrecongnized dataset
        _msg_error("Unrecognized dataset " + dataset)

    #Preparing the ouput file's headers
    header = ["window_size"]

    #Creating the output file
    output_file = open(sys.argv[2] + ".csv", 'w')
    output = Evaluation(output_file, header, dataset.ACTIVITIY_NAMES)
    if "--posterior" in sys.argv:
        posterior_file = open(sys.argv[2] + "_posterior.csv", 'w')
        output.add_prior_prob(posterior_file, header, dataset.ACTIVITIY_NAMES)
    confusion_file = open(sys.argv[2] + ".cm", 'w')
    
    #We first divide the testing and training data
    try:
        test = dataset.obtaining_data()
        training_int_data, testing_int_data = test
    except Exception as exc:
        _msg_error(exc)
    stored_results = np.zeros((len(WINDOW_SIZES),
                               len(dataset.ACTIVITIY_NAMES)+1))
    stored_results_index = 0

    #If necessary we build an Mutual Information matrix
    mi_matrix = None
    if sys.argv[1] == "MI":
        mi_matrix = mi.obtain_mutual_information_matrix(flatten(training_int_data),
                                                        dataset.NUM_EVENTS)

    #We test different window sizes
    for window_size in WINDOW_SIZES:
        #We segment the data
        training_segmented_data = fe.segment_data(training_int_data,
                                                  window_size)
        testing_segmented_data = fe.segment_data(testing_int_data,
                                                 window_size)

        #If necessary we build an Mutual Information Extended matrix
        if sys.argv[1] in ["EMI", "PWC+EMI", "TD+EMI", "PWC+TD+EMI"]:
            mi_matrix = mi.obtain_mutual_information_ext_matrix(training_segmented_data,
                                                                dataset.NUM_EVENTS)

        #Obtain feature vectors
        temp_data = fe.obtain_event_segmentation_data(training_segmented_data,
                                                      feature_vector,
                                                      dataset.NUM_EVENTS,
                                                      mi_matrix)
        training_data, training_class = temp_data
        temp_data = fe.obtain_event_segmentation_data(testing_segmented_data,
                                                      feature_vector,
                                                      dataset.NUM_EVENTS,
                                                      mi_matrix)
        testing_data, testing_class = temp_data

        #We release the segmented_data variables to allow the garbage collector
        #To release memory
        training_segmented_data = 0; testing_segmented_data = 0

        #We create and train the classifier
        classifier = GaussianNB()
        classifier.fit(training_data, training_class)

        #We evaluate and store the result:
        params = [window_size]
        res = output.evaluate_and_store_nb(params, classifier,
                                           training_data, training_class)
        stored_results[stored_results_index, :] = res
        stored_results_index += 1

        #We obtain the confuison matrix
        header = "Window Size: " + str(window_size)
        obtained_testing_class = classifier.predict(testing_data)
        obtained_confusion_matrix = confusion_matrix(testing_class,
                                                     obtained_testing_class,
                                                     len(dataset.ACTIVITIY_NAMES))
        print(header, file = confusion_file)
        print(obtained_confusion_matrix, end = "\n\n\n",
                file = confusion_file)
        
    #We obtain the averages
    averages = ["avg"] + list(np.average(stored_results, 0))
    output._print_to_self(averages, output.file_results)

    #End of execution: we close the output files
    output.close()
    confusion_file.close()
