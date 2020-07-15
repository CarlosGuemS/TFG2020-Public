import sys, re
import numpy as np
from os import path
import random
random.seed(1234)
random_state_fold = np.random.RandomState(1234)
random_state_trees = np.random.RandomState(1234)

#Scikit learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

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
from general import POSIBLE_FEATURE_CONFIG


#Constants
MAX_SAMPLES_TOTAL = 30000
NUM_FOLDS = 10

##STR that indicates usage
USAGE = "Usage: random_forest <feature_mode> <output_name> <data>"

"""
Format of the input configuration: additional features. Possible options:
-BASE: no changes; simple count
-TD: adds time dependency in the count
-EMI: mutual information extened matrix (sensor event window adjency)
-TD+EMI: combination of TD and EMI
-ALL: iteration that serves to test all possible input configurations
"""

if __name__ == "__main__":
    #Reading the inputs
    if len(sys.argv) < 4:
        #Bad usage of the command
        msg_error(USAGE)
    
    #Selecting the dataset
    dataset, dataset_name = load_dataset(sys.argv[3])
    
    #Divide the testing and training data
    try:
        int_data = dataset.obtaining_data()
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
    counter = Counter(len(WINDOW_SIZES) * NUM_FOLDS * (1 + (len(POSIBLE_FEATURE_CONFIG)-1)*(sys.argv[1] == 'ALL')),
                      dataset_name)
    
    
    #Preparing the output files:

    #Confusion matrices
    confusion_matrix_file = out.Confusion_Matrix(sys.argv[2],
                                                dataset.NUM_ACTIVITIES)
    #Accuracy
    accuracy_files = [out.Accuracy_Table(sys.argv[2]+"_"+f+"_ACCURACY")
                      for f in features]
    #Fscore
    fscore_files = [out.Results_Per_Activity(sys.argv[2]+"_"+f+"_FSCORE",
                                            WINDOW_SIZES,
                                            dataset.ACTIVITIY_NAMES)
                    for f in features]

    #Latex files -> Only for ALL
    if sys.argv[1] == "ALL":
        latex_accuracy = out.Latex_Table(len(WINDOW_SIZES)+1, dataset_name)
        latex_fscore = out.Latex_Table(len(WINDOW_SIZES)+1, dataset_name)
    

    #Try different windows
    for window_size in WINDOW_SIZES:
        #We segment the data
        segmented_data = fe.segment_data(int_data, window_size)
        
        #Try different features
        for ff, feature_str in enumerate(features):

            #Configuring the features
            if feature_str == "BASE":
                feature = fe.Features.SIMPLE_COUNT
            elif feature_str == "TD":
                feature = fe.Features.TD_COUNT
            elif feature_str == "EMI":
                feature = fe.Features.MATRIX_COUNT
            elif feature_str == "TD+EMI":
                feature = fe.Features.MATRIX_TD_COUNT
            else:
                print(feature_str)
                msg_error("Feature not identified (this issue should've " +
                        "been checked by now)...")
            
            #Obtain feature vectors
            temp_data = fe.obtain_event_segmentation_data(segmented_data,
                                                          feature,
                                                          dataset.NUM_EVENTS)
            feature_data, feature_class, last_events = temp_data

            #We prepare the arrays to store the results of each K-fold
            temp_accuracy = np.zeros(NUM_FOLDS)
            temp_fscore = np.zeros((NUM_FOLDS, dataset.NUM_ACTIVITIES+1))
            global_predictions = []
            global_true_class = []
            classifier = RandomForestClassifier(n_jobs=-1, random_state=random_state_trees)

            #We must now perform the K folds
            kf = KFold(n_splits=NUM_FOLDS, shuffle = True,
                       random_state=random_state_fold)
            fold_index = 0
            for train_index, test_index in kf.split(feature_data):
                
                #We build an Mutual Information Extended  (if needed)
                emi_data = feature_data
                if feature == fe.Features.MATRIX_COUNT or feature == fe.Features.MATRIX_TD_COUNT:
                    emi_matrix = fe.obtain_mutual_information_ext_matrix(segmented_data,
                                                                         train_index,
                                                                         dataset.NUM_EVENTS)
                    emi_data = fe.apply_sensor_sensor_dependency(feature_data,
                                                                 last_events,
                                                                 emi_matrix,
                                                                 dataset.NUM_EVENTS)

                #We perform the split
                training_data = emi_data[train_index]
                training_class = feature_class[train_index]
                testing_data = emi_data[test_index]
                testing_class = feature_class[test_index]

                #Obtain prior probabilities
                priors = ev.obtain_prior_probabilitites(testing_class,
                                                        dataset.NUM_ACTIVITIES)

                #We run the classifer
                classifier.fit(training_data, training_class)

                #We extract the predictions:
                prediction_classes = ev.obtain_classifier_prediction(dataset.NUM_ACTIVITIES,
                                                                    testing_data,
                                                                    classifier.predict)

                #We update the gobal confusion matrix
                global_predictions.append(prediction_classes)
                global_true_class.append(testing_class)
                
                #Store the accuracy
                temp_accuracy[fold_index] = accuracy_score(testing_class, prediction_classes)
                #Store the fscore
                temp_fscore[fold_index, 1:] = f1_score(testing_class,
                                                       prediction_classes,
                                                       labels=dataset.ACTIVITY_LABELS,
                                                       average=None)
                temp_fscore[fold_index, 0] = (temp_fscore[fold_index, 1:] * priors).sum()

                #We update the fold_index
                fold_index += 1
                 #Printing progress
                counter.checkmarck()

            #We obtain the global values (across all folds)
             #We obtain the global values (across all folds) and store them
            accuracy = temp_accuracy.mean()
            fscore = temp_fscore.mean(axis=0)
            accuracy_files[ff].insert_data(window_size, accuracy)
            fscore_files[ff].store(fscore)

            #We compute the resulting confusion matrix
            global_predictions = np.concatenate(global_predictions, 0)
            global_true_class = np.concatenate(global_true_class, 0)
            global_confusion_matrix = confusion_matrix(global_true_class,
                                                       global_predictions,
                                                       labels = dataset.ACTIVITY_LABELS)

            #We add the result to the confusion matrix file:
            header = "Window Size: " + str(window_size)
            header += "; Feature " + feature_str
            confusion_matrix_file.add_confusion_matrix(header,
                                                       global_confusion_matrix)
            
            #Generate the confusion matrix heat maps
            header = sys.argv[2] + "_Feature_" + feature_str
            header += "_WinSize_" + str(window_size)
            confusion_matrix_file.gen_confusion_matrix_heatmap(header,
                                                               global_confusion_matrix,
                                                               dataset.ACTIVITIY_NAMES,
                                                               True)
        
        #End classifier loop
    
    #End window loop
    
    print("Printing results", dataset_name)
    #Print statisics files
    for ff in range(len(features)):
        #Accuracy
        accuracy_files[ff].print_results()
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
        