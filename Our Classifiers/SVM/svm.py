import sys, re
import numpy as np
from os import path
import random
random.seed(1)

#Scikit learn
from sklearn.svm import SVC
from sklearn.preprocessing import normalize, StandardScaler

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
from general import msg_error, load_dataset, WINDOW_SIZES, Counter


#Constants
SVM_C = 20
SVM_GAMMA = 0.3
MAX_SAMPLES_TOTAL = 20000
MAX_SAMPLES_PER_CLASS = 1200

##STR that indicates usage
USAGE = "Usage: svm <feature_mode> <output_name> <data> [--prior]"

##List of possible feature configurations
POSIBLE_FEATURE_CONFIG = ["BASE", "PWA", "TD", "EMI", "PWA+TD", "PWA+EMI",
                          "TD+EMI", "PWA+TD+EMI"] #There's also ALL


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
    dataset, placeholder_name = load_dataset(sys.argv[3])
    
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
                      placeholder_name)
    
    
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
            #We must also check if we need the svm to calculate probabilitites
            prev_class_pos = -1;
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

            #We check we don't have too many samples:
            if training_class.size > MAX_SAMPLES_TOTAL:
                #We need to limit the amount of samples for SVM to 1000
                limited_training_data = np.empty((0,training_data.shape[1]))
                limited_training_class = np.empty(0)
                for c in range(len(dataset.ACTIVITIY_NAMES)):
                    #Indices of class c
                    class_indices = np.where(training_class == c)[0]
                    #We check if there are more than MAX_SAMPLES_PER_CLASS
                    if class_indices.size > MAX_SAMPLES_PER_CLASS:
                        #If there are more than 100 we must limit the number
                        # of samples to 1000
                        class_indices = random.sample(list(class_indices),
                                                    MAX_SAMPLES_PER_CLASS)
                    #We concatenate the selected samples to the activity set
                    limited_training_data = np.concatenate((limited_training_data,
                                                            training_data[class_indices]),
                                                           0)
                    limited_training_class = np.concatenate((limited_training_class,
                                                             training_class[class_indices]),
                                                            0)
                #We replace the variables
                training_data = limited_training_data
                training_class = limited_training_class
            

            #We standarize and normalize the training data
            scaler = StandardScaler()
            training_data = scaler.fit_transform(training_data)
            normalize(training_data)

            #We run the classifer
            classifier = SVC(SVM_C, 'rbf', gamma=SVM_GAMMA,
                             probability = prev_class_pos != -1)
            classifier.fit(training_data, training_class)

            #We extract the predictions:
            proba_func = None if prev_class_pos == -1 else classifier.predict_proba
            prediction_classes = ev.obtain_classifier_prediction_svm(len(dataset.ACTIVITIY_NAMES),
                                                                     testing_data,
                                                                     prev_class_pos,
                                                                     classifier.predict,
                                                                     proba_func,
                                                                     scaler,
                                                                     normalize)
            
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
            
            #Generate the confusion matrix heat maps
            if window_size == WINDOW_SIZES[0]:
                header = placeholder_name + "_" + feature
                confusion_matrix_file.gen_confusion_matrix_heatmap(header,
                                                                   confusion_matrix,
                                                                   dataset.ACTIVITIY_NAMES,
                                                                   True)
            
            #Printing progress
            counter.checkmarck()

        
        #End classifier loop
    
    #End window loop
    
    print("Printing results...", placeholder_name)
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
        