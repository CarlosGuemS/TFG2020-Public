import sys

import numpy as np

from general import msg_error

#Auxiliary methods
def _round_decimal(value:float, decimals:int):
    """
    Returns a string of given value to a given precision of decimals.
    The value must be found in the range between 0 and 1.
    
    :param value float: value to round
    :param decimals int: number of decimals of precision
    :returns: the string of the decimal at a given position
    :rtype: str
    """

    value_str = str(round(value, decimals))
    return value_str + ("0"*(decimals + 2 - len(value_str)))

def _prev_class_evaluation(initial_class:int, test_data, test_class,
                          position_prev_class:int, predict_function):
    """
    Returns the accuracy score of a scikit-learn classifier (originally
    designed for naive bayes) for the given samples when using the
    prev_class_activity feature.

    :param initial_class:int initial class of the classifier
    :param test_data numpy.array: test partition samples
    :param test_class numpy.array: test_data's classes
    :param position_prev_class int: position of position_prev_class in the
    feature vector
    :param predict_function function: function that makes the prediction by
    the trained classifer
    :returns: the accuracy score and the predicted activities
    :rtype: float, numpy.array
    """

    #We create a copy of the test data
    test_data_copy = np.copy(test_data)
    #We also create a vector where to store the class predictions
    test_prediction = np.zeros(shape=test_class.shape, dtype=np.int32)
    
    #We modify the given copy to adapt its prev_class_evaluation method
    predicted_class = initial_class
    for i in range(test_class.shape[0]):
        #We modify the previous activity
        test_data_copy[i, position_prev_class] = predicted_class

        #We predict the next class
        next_feature = np.resize(test_data_copy[i], (1, test_data_copy.shape[1]))
        predicted_class = test_prediction[i] = np.int32(predict_function(next_feature)[0])

    #Once the classes have been predicted, we check the accuracy
    hits = test_prediction == test_class
    return np.sum(hits) / test_class.shape[0], test_prediction

def obtain_prior_probabilitites(test_samples, num_activities:int):
    """
    Obtain the prior probabilitites of the training samples.
    The classes must be normalized (range from 0 to C-1, where C is the number
    of classes)

    :param test_samples numpy.array: training samples
    :param num_activitites int: number of activities
    """
    prior_probs = np.zeros(shape=(num_activities))
    test_classes = test_samples[:, -1]
    for i in test_classes:
        prior_probs[i] = prior_probs[i] + 1
    return prior_probs / test_classes.shape[0]

##Classes and methods to print results

#Accuracy
class Accuracy_Activity:
    """
    Creates a file storing the accuracies per class

    :param file_name file: file object where to write the results
    :param windows list: windows used
    :param list_activities int: activities used
    """

    def __init__(self, file_name:str, windows:list, list_activities:list):
        self.file_name = file_name
        self.file_prior = None
        self.number_activities = len(list_activities)
        self.values_size = len(windows)+1
        self.data = np.zeros(shape=(self.values_size,
                                    self.number_activities+1))
        self.index = 0
        self.max_index = len(windows)

        #Creation of the header
        self.activities_names = (["Window Size", "Global accuracy"] +
                                ["Activity " + act for act in list_activities])
        self.header = windows.copy()
        self.header += ["Average"]
    
    def evaluate_and_store(self, classifier, testing_data, testing_class,
                           prev_class_pos:int = -1,
                           prev_class_initial_class: int = None):
        """
        Add the values to the array, is posible.

        :param classifier: the naive bayes classifier
        :param testing_data numpy.array: data vectors
        :param testing_class: the class of the data vectors
        :param prev_class_pos: the index where the prev_class attribute is
        found in the feature vector(or -1 if it isn't used)
        :returns: the predicted data
        :rtype: numpy.array
        """
        #We prepare to print the results
        results_to_return = []
        #We select the score depending if it uses prev_class or not
        if prev_class_pos == -1:
            global_score = classifier.score(testing_data, testing_class)
            prediction = np.array(classifier.predict(testing_data),
                                  dtype=np.int32)
        else:
            data = _prev_class_evaluation(prev_class_initial_class,
                                          testing_data, testing_class,
                                          prev_class_pos, classifier.predict)
            global_score, prediction = data
        #We store the global result
        results_to_return.append(global_score)
        
        #We prepare to measure the accuracy for each class
        hits_per_class = [0]*(self.number_activities)
        total_per_class = [0]*(self.number_activities)
        for tr, c in zip(prediction, testing_class):
            #We select data of the appropiate class
            #And measure the amount of times the predicton is correct
            hits_per_class[c] = hits_per_class[c] + int(tr == c)
            total_per_class[c] = total_per_class[c] + 1


        for i in range(self.number_activities):
            #Measure the ratio hits vs misses
            if total_per_class[i]:
                results_to_return.append(hits_per_class[i] / total_per_class[i])
            else:
                #If there's no samples of the given class, the accuracy is 0
                results_to_return.append(0.0)

        if self.index < self.max_index:
            self.data[self.index, :] = results_to_return
            self.index += 1
        else:
            msg_error("Table already full")
        
        return prediction
    
    def print_results(self):
        """
        Prints the results stored in the object
        """
        #We print a warning if the table isn't full yet
        if self.index < self.max_index:
            print("Warning: Table of prior probabilities not full",
                  file=sys.stderr)

        #We obtain the averages
        for i in range(self.data.shape[1]):
            self.data[-1, i] = np.average(self.data[0:-1, i])
        
        #We open the file
        table_file = open(self.file_name + ".csv", 'w')

        #We print the first row
        for x in self.activities_names:
            print(x, end=", ", file=table_file)
        print(file=table_file)

        #We print the rest of the table
        for row in range(self.values_size):
            #First we print the header of the row
            print(self.header[row], end=", ", file = table_file)

            for value in self.data[row, :]:
                print(value, end=", ", file = table_file)

            #Close row
            print(self.data[row, -1], file = table_file)
        
        #We close the file
        table_file.close()
    
    def return_global(self):
        """
        Returns the global accuracy results

        :returns: the global accuracy
        :rtype: numpy.array
        """
        #We print a warning if the table isn't full yet
        if self.index < self.max_index:
            print("Warning: Table of prior probabilities not full",
                  file=sys.stderr)
        
        return self.data[:, 0]

#Prior Probabilities
class Prior_Probabilities:
    """
    Creates a file storing the prior probabilities

    :param file_name str: name of the file where to store the probabilitites
    :param num_activities int: number of columns
    :param header list: header for the table
    """
    def __init__(self, file_name:str, num_activities:int,
                 header:list = [5, 12, 19]):
        self.data = np.zeros(shape=(len(header), num_activities))
        self.index = 0
        #We store other values
        self.header = header
        self.max_index = len(header)
        self.values_size = num_activities
        self.file_name = file_name
    
    def store_result(self, values:list):
        """
        Add the values to the array, is posible.

        :param values list: list of values to store
        """
        if len(values) != self.values_size:
            msg_error("Values are not the correct error")
        elif self.index < self.max_index:
            self.data[self.index, :] = values
            self.index += 1
        else:
            msg_error("Table already full")
    
    def print_results(self):
        """
        Prints the results stored in the object

        """
        #We print a warning if the table isn't full yet
        if self.index < self.max_index:
            print("Warning: Table of prior probabilities not full",
                  file=sys.stderr)
        
        table_file = open(self.file_name + "_prior.csv", 'w')
        for row in range(self.values_size):
            #First we print the header of the row
            print(self.header[row], end=", ", file = table_file)

            for value in self.data[row, :]:
                print(value, end=", ", file = table_file)

            #Close row
            print(self.data[row, -1], file = table_file)
        table_file.close()

#Confusion matrix
class Confusion_Matrix:
    """
    Creates a file storing the confusion matrix

    :param file_name str: name of the file where to store the matrix
    :param number_activities: int: the number of activitites
    """
    def __init__(self, file_name:str, number_activities:int):
        self.file = open(file_name + ".cm", 'w')
        self.number_activities = number_activities
    
    def close(self):
        """
        Close the internal file
        """
        self.file.close()
    
    def add_confusion_matrix(self, header:str, real_class, obtained_class):
        """
        Creates a confusion matrix in the file

        :param real_class numpy.array: the real classes
        :param obtained_class numpy.array: the obtained class
        :returns: the confusion matrix
        :rtype: numpy.array
        """
        #Print the header
        print(header, file=self.file)

        #Create the matrix
        confusion_matrix = np.zeros(shape=(self.number_activities,
                                        self.number_activities),
                                    dtype=np.float64)
        #Count the number of hits
        for real_act, obtained_act in zip(real_class, obtained_class):
            confusion_matrix[real_act, obtained_act] += 1
        #Obtain the averages
        for ii in range(self.number_activities):
            total = np.sum(confusion_matrix[ii, :])
            total = total if total else 1
            confusion_matrix[ii, :] = confusion_matrix[ii,:] / total
        
        #Print the matrix
        print(confusion_matrix, end = "\n\n\n", file = self.file)

#Latex accuracy table
class Accuracy_Table:
    """
    Creates a table to store the results to later be printed as the contents
    of a table in LaTex.

    :param num_cols int: number of columns
    :param header list: header for the table
    
    """
    def __init__(self, num_cols:int, header:list = [5, 12, 19, "Average"]):
        #We create the array
        self.data = np.zeros(shape=(len(header), num_cols))
        self.index = 0
        #We store other values
        self.header = header
        self.max_index = num_cols
        self.values_size = len(header)

    def store_result(self, values:list):
        """
        Add the values to the array, is posible.

        :param values list: list of values to store
        """
        if len(values) != self.values_size:
            msg_error("Values are not the correct size")
        elif self.index < self.max_index:
            self.data[:, self.index] = values
            self.index += 1
        else:
            msg_error("Table already full")
    
    def print_results(self, file_name, bold_option:list = [3],
                      num_decimal:int = 4, split:list = [4,4]):
        """
        Prints the results stored in the object

        :param file_name str: the name of the file to be stored
        :param bold_option list: list of indexes where to print in bold
        :param num_decimal int: number of decimals to print
        :param split int: how to split the table in several tables
        """
        #We print a warning if the table isn't full yet
        if self.index < self.max_index:
            print("Warning: Table of results not full",file=sys.stderr)
        
        #We print an error if the bold options aren't correct
        for bo in bold_option:
            if type(bo) is not int or bo < 0 or bo >= self.values_size:
                msg_error("Error in bold_option in print results: "+ str(bo))
        
        table_file = open(file_name + ".table", 'w')
        first_column = 0
        #We print all tables
        for column_range in split:
            last_column = first_column+column_range-1
            #We print each table
            for row in range(self.values_size):
                #First we print the header of the row
                print(self.header[row], end=" & ", file = table_file)

                for value in self.data[row, first_column:last_column]:
                    val_to_print = _round_decimal(value, num_decimal)
                    #We check if we have to print it as a bold
                    if row in bold_option:
                        val_to_print = "\\textbf{" + val_to_print + "}"
                    #We print the value
                    print(val_to_print, end=" & ", file = table_file)

                #Close row
                val_to_print = _round_decimal(self.data[row, last_column],
                                              num_decimal)
                #We check if we have to print it as a bold
                if row in bold_option:
                    val_to_print = "\\textbf{" + val_to_print + "}"
                print(val_to_print + "\\\\", file = table_file)

                if row == self.values_size-2:
                    print("\\hline", file = table_file)

            #Update the first column
            first_column += column_range
            print(file = table_file)