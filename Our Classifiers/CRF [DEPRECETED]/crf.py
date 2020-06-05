import sys
import numpy as np
from os import path

import pycrfsuite as crf

sys.path.append('../Libraries')
import feature_extraction as fe
import evaluation as ev
from general import msg_error, load_dataset
from output import Confusion_Matrix

sys.path.append('../Datasets')
import Kyoto1, Kyoto2, Kyoto3, Aruba


USAGE = "Usage: crf <output_name> <data>"

if __name__ == "__main__":
    #Reading the inputs
    if len(sys.argv) < 3:
        #Bad usage of the command
        msg_error(USAGE)

    #Selecting the dataset
    dataset, dataset_name = load_dataset(sys.argv[2])

    #Divide the testing and training data
    try:
        int_data_tr, int_data_te = dataset.obtaining_data_continuos()
    except Exception as exc:
        msg_error(exc)
    
    #Obtained the chains of features   
    chained_tr_data = fe.extract_sensor_chains(int_data_tr)
    chained_te_data = fe.extract_sensor_chains(int_data_te)

    #Create the trainer
    trainer = crf.Trainer(verbose=False)
    for fe, cl in chained_tr_data:
        trainer.append(fe, cl)

    #Train the CRF
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'feature.possible_transitions': False
    })
    trainer.train(sys.argv[1]+'.crfsuite')

    #Evaluation: creation of the evaluator
    tagger = crf.Tagger()
    tagger.open(sys.argv[1]+'.crfsuite')

    #For each of the chains of testing features
    prediction_classes = np.empty(0, dtype=int)
    testing_class = np.empty(0, dtype=int)
    for fe, cl in chained_te_data:
        #Add the correct classes to the corresponding array
        cl = np.array(list(map(int, cl)))
        testing_class = np.concatenate((testing_class, cl), 0)

        #Add the prediction
        prediction = np.array(list(map(int, tagger.tag(fe))))
        prediction_classes = np.concatenate((prediction_classes, prediction),
                                            0)

    #We create the confusion matrix to complete the evaluation
    confusion_matrix = ev.obtain_confusion_matrix(dataset.NUM_ACTIVITIES,
                                                  testing_class,
                                                  prediction_classes)
    
    #Creation of the evaluation file
    with open(sys.argv[1] + ".txt", 'w') as output:
        #We print the accuracy
        print("ACCURACY:", ev.obtain_accuracy(confusion_matrix), file=output)
        #We print the precision
        prec = ev.obtain_precision(confusion_matrix)
        print("\nPRECISION", prec[0], file=output)
        print("PER CLASS:", *prec[1:], file=output)
        #We print the recall
        rec = ev.obtain_recall(confusion_matrix)
        print("\nRECALL", rec[0], file=output)
        print("PER CLASS:", *rec[1:], file=output)
        #We print the f-score
        fscore = ev.obtain_fscore(prec[1:], rec[1:])
        print("\nF-SCORE", fscore[0], file=output)
        print("PER CLASS:", *fscore[1:], file=output)
    
    #Creation of the confusion matrix file
    confusion_matrix_file = Confusion_Matrix(sys.argv[1],
                                             dataset.NUM_ACTIVITIES)
    confusion_matrix_file.add_confusion_matrix("", confusion_matrix)
    #Generate the confusion matrix heat maps
    confusion_matrix_file.gen_confusion_matrix_heatmap(sys.argv[1],
                                                       confusion_matrix,
                                                       dataset.ACTIVITIY_NAMES,
                                                       True)
    confusion_matrix_file.close()