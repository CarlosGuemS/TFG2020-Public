import sys, re
import numpy as np
from os import path

#Scikit learn
from sklearn.naive_bayes import GaussianNB

#Preprocessing
sys.path.append('../Libraries')
import feature_extraction as fe

#Output
import evaluation as ev
import output as out

#Datasets
sys.path.append('../Datasets')
import Kyoto1, Kyoto2, Kyoto3, Aruba

#Other
from general import msg_error


#Constants
#WINDOW_SIZES = [5, 12, 19]
WINDOW_SIZES = [5, 12, 19, 26, 33]

##STR that indicates usage
USAGE = "Usage: naive_bayes <feature_mode> <output_name> <data> [--prior]"

##List of possible feature configurations
POSIBLE_FEATURE_CONFIG = ["BASE", "PWA", "TD", "EMI", "PWA+TD", "PWA+EMI",
                          "TD+EMI", "PWA+TD+EMI"] #There's also ALL

##Auxiliary methods
class Counter:
    """
    Class meant to count the progress of the script

    :param number int: number of classifiers
    :param number str: name of the dataset
    """
    def __init__(self, number:int, dataset_name:str):
        self.counter = 0
        self.number = number
        self.dataset_name = dataset_name
    
    def checkmarck(self):
        """
        Updates progress
        """
        self.counter += 1
        print(self.counter, '/', self.number, self.dataset_name)

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
    dataset = sys.argv[3]; placeholder_name = None
    if dataset == "KYOTO1":
        dataset = Kyoto1
        placeholder_name = "OA"
    elif dataset == "KYOTO2":
        dataset = Kyoto2
        placeholder_name = "OAE"
    elif dataset == "KYOTO3":
        dataset = Kyoto3
        placeholder_name = "IwA"
    elif dataset == "ARUBA":
        dataset = Aruba
        placeholder_name = "DLR"
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
    counter = Counter(len(WINDOW_SIZES) * (1 + (len(POSIBLE_FEATURE_CONFIG)-1)*(sys.argv[1] == 'ALL')),
                      sys.argv[3])
    
    
    #Preparing the output files:

    #Confusion matrices
    confusion_matrix_file = out.Confusion_Matrix(sys.argv[2],
                                                len(dataset.ACTIVITIY_NAMES))
    #Accuracy
    accuracy_files = [out.Accuracy_Table(sys.argv[2]+"_"+f+"_ACCURACY")
                      for f in features]
    #Recall 
    recall_files = [out.Results_Per_Activity(sys.argv[2]+"_"+f+"_RECALL",
                                             WINDOW_SIZES,
                                             dataset.ACTIVITIY_NAMES)
                    for f in features]
    #Precision
    precision_files = [out.Results_Per_Activity(sys.argv[2]+"_"+f+"_PRECISON",
                                                WINDOW_SIZES,
                                                dataset.ACTIVITIY_NAMES)
                       for f in features]
    #Fscore
    fscore_files = [out.Results_Per_Activity(sys.argv[2]+"_"+f+"_ACCURACY",
                                            WINDOW_SIZES,
                                            dataset.ACTIVITIY_NAMES)
                    for f in features]
    
    #Propr probabilitites: not always!
    prior_needed = "--prior" in sys.argv
    prior_file = latex_accuracy = latex_fscore = None
    if prior_needed:
        prior_file = out.Prior_Probabilities(sys.argv[2],
                                            len(dataset.ACTIVITIY_NAMES))

    #Latex files -> Only for ALL
    if sys.argv[1] == "ALL":
        latex_accuracy = out.Latex_Table(len(WINDOW_SIZES)+1, placeholder_name)
        latex_fscore = out.Latex_Table(len(WINDOW_SIZES)+1, placeholder_name)
    

    #Try different windows
    for window_size in WINDOW_SIZES:
        #We segment the data
        training_segmented_data = fe.segment_data(training_int_data,
                                                  window_size)
        testing_segmented_data = fe.segment_data(testing_int_data,
                                                 window_size)

        #We build an Mutual Information Extended matrix
        emi_matrix = fe.obtain_mutual_information_ext_matrix(training_segmented_data,
                                                             dataset.NUM_EVENTS)

        #We measure the prior probabilities if needed:
        if prior_needed:
            priors = ev.obtain_prior_probabilitites(training_int_data,
                                                    len(dataset.ACTIVITIY_NAMES))
            prior_file.store_result(priors)
        
        #Try different features
        for ff, feature in enumerate(features):

            #Configuring the features
            prev_class_pos = -1
            if feature == "BASE":
                feature_vector = [fe.Features.SIMPLE_COUNT]
            elif feature == "PWA":
                feature_vector = [fe.Features.PWA, fe.Features.SIMPLE_COUNT]
                prev_class_pos = fe.NUMBER_BASE_FEATURES
            elif feature == "TD":
                feature_vector = [fe.Features.TD_COUNT]
            elif feature == "EMI":
                feature_vector = [fe.Features.MATRIX_COUNT]
            elif feature == "PWA+TD":
                feature_vector = [fe.Features.PWA, fe.Features.TD_COUNT]
                prev_class_pos = fe.NUMBER_BASE_FEATURES
            elif feature == "PWA+EMI":
                feature_vector = [fe.Features.PWA, fe.Features.MATRIX_COUNT]
                prev_class_pos = fe.NUMBER_BASE_FEATURES
            elif feature == "TD+EMI":
                feature_vector = [fe.Features.MATRIX_TD_COUNT]
            elif feature == "PWA+TD+EMI":
                feature_vector = [fe.Features.PWA,
                                  fe.Features.MATRIX_TD_COUNT]
                prev_class_pos = fe.NUMBER_BASE_FEATURES
            else:
                print(feature)
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
            classifier = GaussianNB(var_smoothing=1e-7)
            classifier.fit(training_data, training_class)

            #We extract the predictions:
            prediction_classes = ev.obtain_classifier_prediction(len(dataset.ACTIVITIY_NAMES),
                                                                 testing_data,
                                                                 prev_class_pos,
                                                                 classifier.predict,
                                                                 classifier.predict_proba)
            
            #We build a confusion matrix to measure the quality of results
            confusion_matrix = ev.obtain_confusion_matrix(len(dataset.ACTIVITIY_NAMES),
                                                          testing_class,
                                                          prediction_classes)
            
            
            #We evaluate and store the result of the accuracy, precision
            #and recall
            accuracy = ev.obtain_accuracy(confusion_matrix)
            precision = ev.obtain_precision(confusion_matrix)
            recall = ev.obtain_recall(confusion_matrix)
            fscore = ev.obtain_fscore(precision[1:], recall[1:])

            accuracy_files[ff].insert_data(window_size, accuracy)
            precision_files[ff].store(precision)
            recall_files[ff].store(recall)
            fscore_files[ff].store(fscore)

            #We add the result to the confusion matrix file:
            header = "Window Size: " + str(window_size)
            header += "; Feature " + feature
            confusion_matrix_file.add_confusion_matrix(header,
                                                       confusion_matrix)
            
            #Printing progress
            counter.checkmarck()

        
        #End classifier loop
    
    #End window loop
    
    print("Printing results...", sys.argv[3])
    #Print prior probabilitites (if needed)
    if prior_needed:
        prior_file.print_results()
    #Print statisics files
    for ff in range(len(features)):
        #Accuracy
        accuracy_files[ff].print_results()
        #Precision
        precision_files[ff].print_results()
        #Recall
        recall_files[ff].print_results()
        #Fscore
        fscore_files[ff].print_results()

    #closing confusion matrix file
    confusion_matrix_file.close()
    #Printing Latex files
    if sys.argv[1] == "ALL":
        #Preparing the latex tables
        for ff in range(len(features)):
            latex_accuracy.store_result(accuracy_files[ff].obtain_results())
            latex_fscore.store_result(fscore_files[ff].return_global())
        #Priting the latex files
        latex_accuracy.print_results(sys.argv[2] + "_ACCURACY")
        latex_fscore.print_results(sys.argv[2] + "_FSCORE",
                                   average_list_print= True)
        