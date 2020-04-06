import sys, re
import numpy as np
from os import path

#Scikit learn
from sklearn.naive_bayes import GaussianNB

#Preprocessing
sys.path.append('../Libraries')
import feature_extraction as fe
import discretization as disc

#Output
sys.path.append('../Evaluation')
from evaluation import Evaluation, confusion_matrix

#Datasets
sys.path.append('../Datasets')
import Kyoto

##STR that indicates usage
USAGE = "Usage: naive_bayes <input_config_file> <output_name> <data> [--posterior]"

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

def _string__list_to_int_gen(l:list):
    """
    Given converts a list of string to a generator of int
    
    :params l list: list of string
    :return: generator of int
    :rtype: map object
    """
    return map(lambda x: int(x), l)

def _string__list_to_float_gen(l:list):
    """
    Given converts a list of string to a generator of float
    
    :params l list: list of string
    :return: generator of float
    :rtype: map object
    """
    return map(lambda  x: float(x), l)

"""
Format of the input configuration: different sections, separated by '$$'
Elements within the same section are separated by ','. '\\n' will be ignored.
Sections:
- Discretization (of input)
- Fold: number of participants used in testing instead of training
- Window size
- Feature vector: features in the feature vector used
- Class extract: how to extract the class of a segment
"""
def _obtain_config_parameters(input_config_str:str):
    """
    Given a configuration string obtain the configuration parameters
    :raises ValueError: if the input parameters are not correct

    :param input_config_str str: the string from file with the configuration
    :return: 
    - Discretization (of input). Either a string or none
    - Fold: number of participants used in testing instead of training
    - Window size
    - Feature vector: features in the feature vector used. Return as a list.
    - Class extraction: how to extract the class of a segment
    """

    input_config_str = re.sub("\n", "", input_config_str)
    input_config = re.split("\$\$", input_config_str)

    #print(input_config)

    #Raises exception if the number of fields isn't correct
    if len(input_config) != 5:
        raise ValueError("The number of categories in the input configuration"
                         + " isn't 5 (" + str(len(input_config)) + ")")

    #Discretization
    discretization = input_config[0] if input_config[0] else "NONE"

    #Number of fold(s) to use
    if not len(input_config[1]):
        raise ValueError("We require at least 1 value in fold")
    folds_str = re.split(',', input_config[1])
    folds = _string__list_to_int_gen(folds_str)
    
    #Number of window sizes to use
    if not len(input_config[2]):
        raise ValueError("We require at least 1 value in windows size")
    window_sizes_str = re.split(',', input_config[2])
    window_sizes = list(_string__list_to_int_gen(window_sizes_str))

    #Features
    feature_vector = re.split(',', input_config[3])
    
    #Class extraction
    class_extraction = input_config[4]
    if not input_config:
        _msg_error("Class extraction is required")
    
    return (discretization, folds, window_sizes, feature_vector,
            class_extraction)

#Main
if __name__ == "__main__":
    #Reading the inputs
    if len(sys.argv) < 4:
        _msg_error(USAGE)
    try:
        input_file = open(path.abspath(sys.argv[1]), "r")
        input_config_str = input_file.read()
        input_file.close()
    except:
        _msg_error("input file " + path.abspath(sys.argv[1]) 
                   +  " doesn't exist")
    
    try:
        input_config = _obtain_config_parameters(input_config_str)
    except ValueError as exc:
        _msg_error(exc.args)
    discretization = input_config[0]
    folds = input_config[1]
    window_sizes = input_config[2]
    feature_vector = input_config[3]
    class_extraction = input_config[4]

    #Selecting the dataset
    dataset = sys.argv[3]
    if dataset == "KYOTO":
        dataset = Kyoto
    else:
        #Unrecongnized dataset
        _msg_error("Unrecognized dataset " + dataset)

    #Preparing the ouput file's headers
    header = ["fold", "window_size"];

    #Creating the output file
    output_file = open(sys.argv[2] + ".csv", 'w')
    output = Evaluation(output_file, header, dataset.ACTIVITIY_NAMES, True)
    if "--posterior" in sys.argv:
        posterior_file = open(sys.argv[2] + "_posterior.csv", 'w')
        output.add_prior_prob(posterior_file, header, dataset.ACTIVITIY_NAMES)
    confusion_file = open(sys.argv[2] + ".cm", 'w')
    
    #We first divide the testing and training data
    for fold in folds:
        if fold <= 0 or fold > len(dataset.PARTICIPANTS):
            _msg_error("Fold not appropiate for the selected dataset")
        
        try:
            temp = dataset.obtain_participants_data(fold)
            training_int_data, testing_int_data = temp
        except Exception as exc:
            _msg_error(exc)

        #Perform discretization
        if discretization == "NONE":
            pass
        elif discretization == "SET ALL TO 1":
            for participiant_data in training_int_data:
                disc.set_all_to_1(participiant_data)
            for participiant_data in testing_int_data:
                disc.set_all_to_1(participiant_data)
        else:
            _msg_error("Unrecognized discretization: " + discretization)
        #Obtain windows
        for window_size in window_sizes:
            #Obtain feature vectors
            temp_data = fe.obtain_event_segmentation_data(training_int_data,
                                                          window_size,
                                                          feature_vector,
                                                          class_extraction,
                                                          dataset.NUM_EVENTS)
            training_data = temp_data[0]; training_class = temp_data[1]
            temp_data = fe.obtain_event_segmentation_data(testing_int_data,
                                                          window_size,
                                                          feature_vector,
                                                          class_extraction,
                                                          dataset.NUM_EVENTS)
            testing_data = temp_data[0]; testing_class = temp_data[1]

            #We create and train the classifier
            classifier = GaussianNB()
            classifier.fit(training_data, training_class)

            #We evaluate the result:
            params = [fold, window_size]
            output.evaluate_and_store_nb(params, classifier, training_data,
                                         training_class)
            #We obtain the confuison matrix
            header = "Fold: " + str(fold)
            header += "; Window Size: " + str(window_size)
            obtained_testing_class = classifier.predict(testing_data)
            obtained_confusion_matrix = confusion_matrix(testing_class,
                                                         obtained_testing_class,
                                                         len(dataset.ACTIVITIES))
            print(header, file = confusion_file)
            print(obtained_confusion_matrix, end = "\n\n\n",
                  file = confusion_file)

    #End of execution: we close the output files
    output.close()
    confusion_file.close()
