import sys, re
import numpy as np
from os import path

#Scikit learn
from mixed_naive_bayes import MixedNB

#Preprocessing
sys.path.append('../Libraries')
import feature_extraction as fe
import mutual_information as mi

#Output
import evaluation as ev

#Datasets
sys.path.append('../Datasets')
import Kyoto1, Kyoto2, Kyoto3, Aruba

#Other
from general import msg_error


#Constants
WINDOW_SIZES = [5, 12, 19]

##STR that indicates usage
USAGE = "Usage: naive_bayes <feature_mode> <output_name> <data> [--prior]"

##List of possible feature configurations
POSIBLE_FEATURE_CONFIG = ["BASE", "PWA", "TD", "EMI", "PWA+TD", "PWA+EMI",
                          "TD+EMI", "PWA+TD+EMI"] #There's also ALL

##Auxiliary methods
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
-MI_EXT: mutual information extened matrix (sensor event window adjency)
-ALL: iteration that serves to test all possible input configurations
"""

if __name__ == "__main__":
    #Reading the inputs
    if len(sys.argv) < 4:
        #Bad usage of the command
        msg_error(USAGE)
    
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
        msg_error("Unrecognized dataset " + dataset)
    
    #Divide the testing and training data
    try:
        data = dataset.obtaining_data()
        training_int_data, testing_int_data = data
    except Exception as exc:
        msg_error(exc)
    
    #Selecting which features to test
    if sys.argv[1] == "ALL":
        features = POSIBLE_FEATURE_CONFIG.copy()
    elif sys.argv[1] in POSIBLE_FEATURE_CONFIG:
        features = [sys.argv[1]]
    else:
        #Unkown feature
        msg_error("Unkown feature: " + sys.argv[1])
    
    
    #Preparing the output files
    confusion_matrix_file = ev.Confusion_Matrix(sys.argv[2],
                                                len(dataset.ACTIVITIY_NAMES))
    results_per_class_file = [ev.Accuracy_Activity(sys.argv[2]+"_"+f,
                                                   WINDOW_SIZES,
                                                   dataset.ACTIVITIY_NAMES)
                              for f in features]
    prior_needed = "--prior" in sys.argv
    prior_file = latex_file = None
    if prior_needed:
        prior_file = ev.Prior_Probabilities(sys.argv[2],
                                            len(dataset.ACTIVITIY_NAMES))
    if sys.argv[1] == "ALL":
        latex_file = ev.Accuracy_Table(len(POSIBLE_FEATURE_CONFIG))
    
    #Try different windows
    for window_size in WINDOW_SIZES:
        #We segment the data
        training_segmented_data = fe.segment_data(training_int_data,
                                                  window_size)
        testing_segmented_data = fe.segment_data(testing_int_data,
                                                 window_size)

        #We build an Mutual Information Extended matrix
        emi_matrix = mi.obtain_mutual_information_ext_matrix(training_segmented_data,
                                                             dataset.NUM_EVENTS)

        #We measure the prior probabilities if needed:
        if prior_needed:
            priors = ev.obtain_prior_probabilitites(training_int_data,
                                                    len(dataset.ACTIVITIY_NAMES))
            prior_file.store_result(priors)
        
        #Try different features
        for ff in range(len(features)):

            #Configuring the features
            categorical_features = [0]
            max_categories=[7]
            prev_class_pos = -1
            if features[ff] == "BASE":
                feature_vector = ["SIMPLE COUNT SENSOR"]
            elif features[ff] == "PWA":
                feature_vector = ["PREV CLASS LAST EVENT",
                                  "SIMPLE COUNT SENSOR"]
                #PREV CLASS LAST EVENT is discrete, so we have to set it up as one
                categorical_features.append(fe.NUMBER_BASE_FEATURES)
                max_categories.append(len(dataset.ACTIVITIY_NAMES)+1)
                prev_class_pos = fe.NUMBER_BASE_FEATURES
            elif features[ff] == "TD":
                feature_vector = ["TIME DEPEDENCY COUNT SENSOR"]
            elif features[ff] == "EMI":
                feature_vector = ["MATRIX COUNT SENSOR"]
            elif features[ff] == "PWA+TD":
                feature_vector = ["PREV CLASS LAST EVENT",
                                "TIME DEPEDENCY COUNT SENSOR"]
                categorical_features.append(fe.NUMBER_BASE_FEATURES)
                max_categories.append(len(dataset.ACTIVITIY_NAMES)+1)
                prev_class_pos = fe.NUMBER_BASE_FEATURES
            elif features[ff] == "PWA+EMI":
                feature_vector = ["PREV CLASS LAST EVENT",
                                  "MATRIX COUNT SENSOR"]
                categorical_features.append(fe.NUMBER_BASE_FEATURES)
                max_categories.append(len(dataset.ACTIVITIY_NAMES)+1)
                prev_class_pos = fe.NUMBER_BASE_FEATURES
            elif features[ff] == "TD+EMI":
                feature_vector = ["TIME DEPEDENCY MATRIX COUNT SENSOR"]
            elif features[ff] == "PWA+TD+EMI":
                feature_vector = ["PREV CLASS LAST EVENT",
                                  "TIME DEPEDENCY MATRIX COUNT SENSOR"]
                categorical_features.append(fe.NUMBER_BASE_FEATURES)
                max_categories.append(len(dataset.ACTIVITIY_NAMES)+1)
                prev_class_pos = fe.NUMBER_BASE_FEATURES
            else:
                print(features[ff])
                msg_error("Feature not identified (this issue should've " +
                        "been checked by now)...")
            
            #Obtain feature vectors
            temp_data = fe.obtain_event_segmentation_data(training_segmented_data,
                                                          feature_vector,
                                                          dataset.NUM_EVENTS,
                                                          len(dataset.ACTIVITIY_NAMES),
                                                          emi_matrix)
            training_data, training_class = temp_data
            temp_data = fe.obtain_event_segmentation_data(testing_segmented_data,
                                                          feature_vector,
                                                          dataset.NUM_EVENTS,
                                                          len(dataset.ACTIVITIY_NAMES),
                                                          emi_matrix)
            testing_data, testing_class = temp_data

            #We run the classifer
            classifier = MixedNB(categorical_features=categorical_features,
                                 max_categories=max_categories)
            classifier.fit(training_data, training_class)

            #We evaluate and store the result of the Accuracy file:
            obtained_testing_class = results_per_class_file[ff].evaluate_and_store(classifier,
                                                                                   training_data,
                                                                                   training_class,
                                                                                   prev_class_pos,
                                                                                   len(dataset.ACTIVITIY_NAMES))

            #We add the result to the confusion matrix file:
            header = "Window Size: " + str(window_size)
            header += "; Feature " + features[ff]
            confusion_matrix_file.add_confusion_matrix(header, testing_class,
                                                       obtained_testing_class)
        
        #End classifier loop
    
    #End window loop
    
    print("Priting results:")
    #Print prior probabilitites (if needed)
    if prior_needed:
        prior_file.print_results()
    #Print accuracy files
    for ff in range(len(features)):
        results_per_class_file[ff].print_results()
    #closing confusion matrix file
    confusion_matrix_file.close()
    #Printing latex table
    if sys.argv[1] == "ALL":
        for ff in range(len(features)):
            latex_file.store_result(results_per_class_file[ff].return_global())
        latex_file.print_results(sys.argv[2])
        