import sys, re
import numpy as np
from os import path

#Scikit learn
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB

#Preprocessing
sys.path.append('../Libraries')
import feature_extraction as fe
from intermediate_data import obtain_participants_data
import discretization as disc

#Output
sys.path.append('../Evaluation')
from evaluation import Evaluation

#Datasets
sys.path.append('../Datasets')
import Kyoto

##STR that indicates usage
USAGE = "Usage: naive_bayes <input_config_file> <output_name> <data>"

##Auxiliary methods
def _msg_error(msg:str):
    """
    Prints an error in the stderr output and ends the execution prematurely

    :param msg str: error message
    :rtype: None
    """
    print(msg, file=sys.stderr)
    try:
        output_file.close()
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
- Type of naice bayes to use (GAUSS, MULTINOMIAL, CATEGORIAL)
- Fold: number of participants used in testing instead of training
- Window size
- Alpha (for multinomial only, otherwise leave empty): alpha parameter used in
the laplace smoothing
- Number of categories (for categorical only, otherwise leave empty):
discretize vectors to the given number of categories. Categories will be
built from training data to have roughly the same number of elements in
it. Returned as a list
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
    - Type of naive bayes to use (gauss, multinomial, categorical)
    - Fold: number of participants used in testing instead of training
    - Window size
    - Alpha: alpha parameter used in the laplace smoothing in multinomial
    naive bayes
    - Number of categories (for categorical only, otherwise leave empty):
    discretize vectors to the given number of categories. Categories will be
    built from training data to have roughly the same number of elements in
    it. Returned as a list
    - Feature vector: features in the feature vector used. Return as a list.
    - Class extraction: how to extract the class of a segment
    """

    input_config_str = re.sub("\n", "", input_config_str)
    input_config = re.split("\$\$", input_config_str)

    #print(input_config)

    #Raises exception if the number of fields isn't correct
    if len(input_config) != 8:
        raise ValueError("The number of categories in the input configuration"
                         + " isn't 8 (" + str(len(input_config)) + ")")

    #Discretization
    discretization = input_config[0] if input_config[0] else "NONE"
    
    #Type of naive bayes to use
    type_nb = input_config[1]
    if not type_nb:
        raise ValueError("The type of naive_bayes wasn't especified")

    #Number of fold(s) to use
    if not len(input_config[2]):
        raise ValueError("We require at least 1 value in fold")
    folds_str = re.split(',', input_config[2])
    folds = _string__list_to_int_gen(folds_str)
    
    #Number of window sizes to use
    if not len(input_config[3]):
        raise ValueError("We require at least 1 value in windows size")
    window_sizes_str = re.split(',', input_config[3])
    window_sizes = list(_string__list_to_int_gen(window_sizes_str))

    #Alpha parameter
    if not len(input_config[4]):
        if type_nb == "MULTINOMIAL" or type_nb == "CATEGORICAL":
            ##We only give the warning in multinomial is used
            print("Warning: default value of alpha used (1.0):", file =sys.stderr)
        alphas = [1.0]
    else:
        alphas_str = re.split(',', input_config[4])
        alphas = list(_string__list_to_float_gen(alphas_str))

    #Features
    feature_vector = re.split(',', input_config[6])

    #Discretizatoin of features
    try:
        number_categories = int(input_config[5])
    except:
        if type_nb == "CATEGORICAL":
            _msg_error("Number of categories is required for a categorial classifier")
        number_categories = None
    
    #Class extraction
    class_extraction = input_config[7]
    if not input_config:
        _msg_error("Class extraction is required")
    
    return (discretization, type_nb, folds, window_sizes, alphas,
           number_categories, feature_vector, class_extraction)

##Clasificadores

def _gauss_classifier(training_data, training_class, alpha:float = None):
    """
    Gauss classifier

    :param training_data: training data
    :param training_class: training classes
    :param alpha float: ingored in this classifier
    :return: trained classifier
    """
    classifier = GaussianNB();
    classifier.fit(training_data, training_class)
    return classifier

def _multinomial_classifier(training_data, training_class, alpha: float):
    """
    Multinomial classifier

    :param training_data: training data
    :param training_class: training classes
    :param alpha float: alpha parameter
    :return: trained classifier
    """
    classifier = MultinomialNB(alpha)
    classifier.fit(training_data, training_class)
    return classifier


def _categorical_classifier(training_data, training_class, alpha: float):
    """
    Categorial classifier

    :param training_data: training data
    :param training_class: training classes
    :param alpha float: alpha parameter
    :return: trained classifier
    """
    classifier = CategoricalNB(alpha)
    classifier.fit(training_data, training_class)
    return classifier

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
    type_nb = input_config[1]
    folds = input_config[2]
    window_sizes = input_config[3]
    alphas = input_config[4]
    number_categories = input_config[5]
    feature_vector = input_config[6]
    class_extraction = input_config[7]

    #Selecting the dataset
    dataset = sys.argv[3]
    if dataset == "KYOTO":
        dataset = Kyoto
    else:
        #Unrecongnized dataset
        _msg_error("Unrecognized dataset " + dataset)

    #Preparing the ouput file's headers
    header = ["fold", "window_size"]

    #Selecting the classifier:
    if type_nb == "GAUSS":
        call_classifier = _gauss_classifier
    elif type_nb == "MULTINOMIAL":
        header += ["alpha"]
        call_classifier = _multinomial_classifier
    elif type_nb == "CATEGORICAL":
        header += ["alpha"]
        call_classifier = _categorical_classifier
    else:
        _msg_error("Type of naive Bayes not recognized")

    #Creating the output file
    output_file = open(sys.argv[2] + ".csv", 'w')
    output = Evaluation(output_file, header, dataset.ACTIVITIY_NAMES, True)
    
    #We first divide the testing and training data
    for fold in folds:
        if fold <= 0 or fold > len(dataset.PARTICIPANTS):
            _msg_error("Fold not appropiate for the selected dataset")
        training_participants = dataset.PARTICIPANTS[:-fold]
        testing_participants = dataset.PARTICIPANTS[-fold:]
        
        training_int_data = obtain_participants_data(training_participants,
                                                 dataset.FORMATED_DATA_PATH)
        testing_int_data = obtain_participants_data(testing_participants,
                                                dataset.FORMATED_DATA_PATH)
        
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

            #If the classifier is categorial we must discretize the fetures
            if number_categories:
                discretization_func = disc.obtain_categorization_function(training_data,
                                                                          number_categories)
                discretization_func(training_data)
                discretization_func(training_data)

            #We check the values of alpha
            for alpha in alphas:
                #We obtain the trained classifier
                classifier = call_classifier(training_data, training_class,
                                             alpha)
                #We obtained the predicited activities
                #predicited_activities = classifier.predict(testing_data)
                #print(classifier.score(testing_data, testing_class))
                params = [fold, window_size]
                if type_nb == "MULTINOMIAL" or type_nb == "CATEGORICAL":
                    params += [alpha]
                output.evaluate_and_store_nb(params, classifier,
                                             training_data, training_class)

    #End of execution: we close the outputfile
    output_file.close()
