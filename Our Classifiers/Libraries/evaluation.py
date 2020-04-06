import sys

import numpy as np

def _msg_error(msg:str):
    """
    Prints an error in the stderr output and ends the execution prematurely

    :param msg str: error message
    :rtype: None
    """
    print(msg, file=sys.stderr)
    sys.exit(1)

def _recall(obtained_results, real_results, number_activities:int):
    """
    Returns the recall

    :param obtained_results numpy.array: the class obtained by the classifier
    :param real_results numpy.array: the real class
    :param number_activities int: number activities
    :returns: the recall score, absolute and per class
    :rtype: float, list
    """

    total_recall = 0
    activity_recall = [0] * (number_activities+1)
    activity_num = [0] * (number_activities+1)
    for obtained_result, real_result in zip(obtained_results, real_results):
        if obtained_result == real_result:
             total_recall += 1
             activity_recall[real_result] +=1
        activity_num[real_result] += 1
    total_recall /= len(obtained_results)
    activity_recall = list(activity_recall[i]/len(activity_num)
                             for i in range(1,len(activity_num)))

    return total_recall, activity_recall

def _accuracy(obtained_results, real_results, number_activities:int):
    """
    Returns the accuracy

    :param obtained_results numpy.array: the class obtained by the classifier
    :param real_results numpy.array: the real class
    :param number_activities int: number activities
    :returns: the accuracy score, absolute and per class
    :rtype: float, list
    """

    total_accuracy = 0
    activity_accuracy = [0] * (number_activities+1)
    activity_num = [0] * (number_activities+1)
    for obtained_result, real_result in zip(obtained_results, real_results):
        if obtained_result == real_result:
             total_accuracy += 1
             activity_accuracy[obtained_result] +=1
        activity_num[obtained_result] += 1
    total_accuracy /= len(obtained_results)
    activity_accuracy = list(activity_accuracy[i]/len(activity_num)
                             for i in range(1,len(activity_num)))
    return total_accuracy, activity_accuracy

def fscore(obtained_results, real_results, number_activities:int,
           total_accuracy:float = None, activity_accuracy:list = None):
    """
    Returns the (balanced) fscore

    :param obtained_results numpy.array: the class obtained by the classifier
    :param real_results numpy.array: the real class
    :param number_activities int: number activities
    :param total_accuracy float: accuracy to use, if it has been obtained
    before
    :param activity_accuracy float: activity accuracy, it it has been obtained
    before
    :returns: the fscore score, absolute and per class
    :rtype: float, list
    """

    if not (total_accuracy and activity_accuracy):
        total_accuracy, activity_accuracy = _accuracy(obtained_results,
                                                     real_results,
                                                     number_activities)
    total_recall, activity_recall = _recall(obtained_results, real_results,
                                           number_activities)
    
    total_fscore = 2 * total_accuracy * total_recall
    total_fscore /= total_accuracy + total_fscore
    activity_fscore = np.zeros(number_activities)
    i = 0
    for accuracy, recall in zip(activity_accuracy, activity_recall):
        #If either the accuracy or the recall ==0, the fscore is 0 automatically
        try:
            activity_fscore[i] = 2 * accuracy * recall / (accuracy + recall)
        except ZeroDivisionError:
            activity_fscore[i] = 0
        i+=1
    return total_fscore, activity_fscore


class Evaluation:
    """
        Creates an object with an associated file. The file will be used to
        obtain all the results. Parameters and number of activities is to be
        passed to create the header of the file.

        :param file_object file: file object where to write the results
        :param parameters list: Parameters used
        :param list_activities int: activities used
        """

    def __init__(self, file_object, parameters:list,
                 list_activities:list):
        
        self.file_results = file_object
        self.file_prior = None
        self.number_activities = len(list_activities)

        #Creation of the header
        activities_names = ["Activity " + act for act in list_activities]
        header = parameters.copy()
        header += ["Total accuracy"]
        header += ["Accuracy " + act for act in activities_names]
        self._print_to_self(header, self.file_results)

    
    def add_prior_prob(self, file_object, parameters:list,
                       list_activities:list):
        """
        Add a file to output the prior probabilitites for each class with the
        given classifier

        :param file_object file: file object where to write the prior
        probabilitites
        :param parameters list: Parameters used
        :param list_activities int: activities used
        """
        self.file_prior = file_object
        header = parameters.copy()
        header += ["Prior probability " + act for act in list_activities]
        self._print_to_self(header, self.file_prior)
    

    def evaluate_and_store(self, params:list, obtained_results, real_results):
        """
        Given the results of the classifiers and the real results, evaluate
        the accuracy and fscore of both the classifier as a whole and by
        each individual activity

        :param params list: list of parameters used
        :param obtained_results numpy.array: the class obtained by the
        classifier
        :param real_results numpy.array: the real class
        """
        total_accuracy, activity_accuracy = _accuracy(obtained_results,
                                                     real_results,
                                                     self.number_activities)
        total_fscore, activity_fscore = fscore(obtained_results, real_results,
                                               self.number_activities,
                                               total_accuracy,
                                               activity_accuracy)                                            
        results_to_print = [total_accuracy, total_fscore]
        results_to_print += activity_accuracy
        results_to_print += list(activity_fscore)
        self._print_to_self(results_to_print, self.file_results)

    def evaluate_and_store_nb(self, params:list, classifier, testing_data,
                              testing_class):
        """
        Given the results of the classifiers and the real results, evaluate
        the accuracy and fscore of both the classifier as a whole and by
        each individual activity. Naive Bayes version

        :param params list: list of parameters used
        :param classifier: the naive bayes classifier
        :param testing_data numpy.array: data vectors
        :param testing_class: the class of the data vectors
        :returns: the stored data
        :rtype: numpy.array
        """

        #We prepare to print the results
        results_to_return = []
        results_to_return.append(classifier.score(testing_data, testing_class))
        for i in range(1, self.number_activities+1):
            #We select data of the appropiate class
            selected_data = np.array(list(tr
                                          for tr, c in zip(testing_data, testing_class)
                                          if c == i))
            if len(selected_data):
                selected_data_class = np.zeros(len(selected_data)) + i
                results_to_return.append(classifier.score(selected_data,
                                                          selected_data_class))
            else:
                results_to_return.append(0.0);
        
        results_to_print = params.copy() + results_to_return.copy()
        self._print_to_self(results_to_print, self.file_results)

        #We check if we also have to print the prior probability
        if self.file_prior: 
            results_to_print = params.copy()+classifier.class_prior_.tolist()
            self._print_to_self(results_to_print, self.file_prior)
        
        return np.array(results_to_return)


    def _print_to_self(self, list_to_print: list, file_object):
        """
        Given a list, print it in the file according to csv formar

        :param list_to_print: the list to print
        :param file file: file to print
        """
        for r in list_to_print[:-1]:
            print(r, end=',', file = file_object)
        print(list_to_print[-1], file= file_object)
    
    def close(self):
        """
        Closes the open files
        """
        self.file_results.close()
        if self.file_prior:
            self.file_prior.close()

##Other methods
def confusion_matrix(real_class, obtained_class, number_activities:int):
    """
    Creates a confusion matrix

    :param real_class numpy.array: the real classes
    :param obtained_class numpy.array: the obtained class
    :param number_activities int: number of activities
    :returns: the confusion matrix
    :rtype: numpy.array
    """
    confusion_matrix = np.zeros(shape=(number_activities, number_activities), dtype=np.float64)
    real_class = real_class-1; obtained_class = obtained_class-1;
    for real_act, obtained_act in zip(real_class, obtained_class):
        confusion_matrix[real_act, obtained_act] += 1
    for ii in range(number_activities):
        total = np.sum(confusion_matrix[ii, :])
        total = total if total else 1
        confusion_matrix[ii, :] = confusion_matrix[ii,:] / total
    return confusion_matrix
