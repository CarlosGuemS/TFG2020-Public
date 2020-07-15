import sys

import numpy as np
import matplotlib.pyplot as plt

from itertools import product

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

##Classes and methods to print results

#Accuracy
class Accuracy_Table:
    """
    Creates a file to store the accuracies per class

    :param file_name file: file name where to write the results
    """

    header = ["Window Sizes", "Accuracy"]

    def __init__(self, file_name:str):
        self.file_name = file_name
        self.data = []
    
    def insert_data(self, window:int, accuracy:float):
        """
        Adds more data to the table

        :param window int: window size
        :param accuracy float: measured accuracy
        """
        self.data.append((window, accuracy))
    
    def obtain_average(self):
        """
        Returns the average of the measured results
        """
        return sum(val for window, val in self.data) / len(self.data)

    def obtain_results(self):
        """
        Obtains all results
        """
        return np.array([x for w,x in self.data] + [self.obtain_average()])
    
    def print_results(self):
        """
        Prints the results
        """
        #Open file
        table_file = open(self.file_name + ".csv", 'w')
        #Print the headers
        print(*self.header, sep=",", file = table_file)
        #Print the values
        for win, val in self.data:
            print(win, val, sep=",", file = table_file)
        #Print the average
        print("Average", self.obtain_average(), sep=',', file=table_file)
        #Close file
        table_file.close()
        

#Precision/Recall/Fscore
class Results_Per_Activity:
    """
    Creates a file store some result globally or class

    :param file_name file: file name where to write the results
    :param windows list: windows used
    :param list_activities int: activities used
    """

    def __init__(self, file_name:str, windows:list, list_activities:list):
        self.file_name = file_name
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
    
    def store(self, values):
        """
        Add the values to the array, is posible.

        :param values numpy.array: the values to introduce
        """
        #We check if the table is full
        if self.index >= self.max_index:
            msg_error("Table already full")
        
        #We check the rows and colums are correct
        if values.shape[0] != self.number_activities+1:
            msg_error("Incorrect values shape")
        
        #Everything is in order, we insert the value and increase the index
        self.data[self.index, :] = values
        self.index += 1
    
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
        print(*self.activities_names, sep=", ", file=table_file)

        #We print the rest of the table
        for row in range(self.values_size):
            #First we print the header of the row
            print(self.header[row], end=", ", file = table_file)

            #We print the row
            print(*self.data[row, :], sep=", ", file = table_file)
        
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
    
    def add_confusion_matrix(self, header:str, confusion_matrix,
                             normalize:bool = False):
        """
        Stores a given confusion matrix in the file

        :param header str: header for the confusion matrix
        :param confusion_matrix: a, not normalized, confusion matrix
        :param normalize bool: indicates if the matrix is to be normalized
        (the sum rows equals to 1)
        """
        #Print the header
        print(header, file=self.file)
        
        #Obtain the averages (normalizes it)
        if normalize:
            for ii in range(self.number_activities):
                total = np.sum(confusion_matrix[ii, :])
                total = total if total else 1
                confusion_matrix[ii, :] = confusion_matrix[ii,:] / total
        
        #Print the matrix
        print(confusion_matrix, end = "\n\n\n", file = self.file)
    
    def gen_confusion_matrix_heatmap(self, file_name:str, confusion_matrix,
                                     classes:list, normalize:bool = False):
        """
        Generates a heatmap of the given confusion matrix in a file with
        the given name in the current directory

        :param file_name str: the name of the file to store
        :param confusion_matrix numpy.array: the confusion matrix
        :param classes list list: list with the name of the classes
        :param normalize bool: indicates if the matrix is to be normalized
        (the sum rows equals to 1)
        """

        #First we normalize (if needed):
        if normalize:
            confusion_matrix = confusion_matrix.astype(float)
            for ii in range(self.number_activities):
                total = np.sum(confusion_matrix[ii, :])
                total = total if total else 1
                confusion_matrix[ii, :] = confusion_matrix[ii,:] / total

        

        #We create the figure
        fig, ax = plt.subplots(figsize = (8, 8))
        im = ax.imshow(confusion_matrix, cmap="magma")

        #We add the labels on the sides
        ax.set_ylabel('True class')
        ax.set_xlabel('Predicted class')

        #We plot the axes
        num_ticks = np.arange(len(classes))
        ax.set_xticks(num_ticks); ax.set_yticks(num_ticks)
        ax.set_xticklabels(classes); ax.set_yticklabels(classes)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        #We place the text within the matrix
        textcolors=("white", "black")
        threshold = im.norm(confusion_matrix.max()) / 2.0
        textsize = "large" if len(classes) > 8 else "x-large"
        for i, j in product(num_ticks, repeat=2):
            color = textcolors[int(im.norm(confusion_matrix[i, j]) > threshold)]
            ax.text(j, i, _round_decimal(confusion_matrix[i, j], 3),
                    weight="semibold", fontsize=textsize, ha="center",
                    va="center", color=color)
        
        #Save file
        fig.tight_layout()
        plt.savefig(file_name + ".pdf", format='pdf')


#Latex accuracy table
class Latex_Table:
    """
    Creates a table to store the results to later be printed as the contents
    of a table in LaTex.

    :param entries_per_row list: number of entries per row
    :param dataset_name str: name of the dataset
    """

    #We lay out the constants: headers and similar
    row_headers = ["\multirow{4}{*}{\PLACEHOLDER} & Base",
                   "\cline{2-8} & Base + TD", "\cline{2-8} & Base + SD",
                   "\cline{2-8} & Base + TD + SD"]
    
    num_rows = 4

    def __init__(self, entries_per_row:int, dataset_name:str):
        #Array to store the data
        self.data = np.zeros(shape=(self.num_rows, entries_per_row))
        #Indices (to know how to store the array)
        self.index = 0
        #Number of entries per row
        self.entries_per_row = entries_per_row
        #We change the dataset name
        self.row_headers[0] = self.row_headers[0].replace("PLACEHOLDER",
                                                          dataset_name,
                                                          1)

    def store_result(self, values:list):
        """
        Add the values to the array, is posible.

        :param values list: list of values to store
        """
        if len(values) != self.entries_per_row:
            msg_error("Values are not the correct size")

        if self.index < self.num_rows:
            self.data[self.index, :] = values
            self.index += 1
        else:
            msg_error("Table already full")

    def print_results(self, file_name:str, num_decimal:int = 4,
                      average_list_print:bool = False):
        """
        Prints the results stored in the object

        :param file_name str: the name of the file to be stored
        :param num_decimal int: number of decimals to print (DEPRECATED)
        :param average_list_print: indicates if the total averages are printed
        as a separate list. Used to generate graphs in matplotlib
        """
        #We ouput an error if the file isn't full yet
        if self.index != self.num_rows:
            msg_error("Error: Table of results not full")
        
        #We open the file:
        table_file = open(file_name + ".table", 'w')
        
        #We print the row
        for ii in range(self.num_rows):
            #Row header
            print(self.row_headers[ii], end='', file=table_file)
            #Row content
            for elem in self.data[ii, :]:
                print(" & ", "%.4f" % elem, sep='', end='', file=table_file)
            #Row end
            print(' \\\\', file=table_file)
        
        #We print the horizontal line diving this dataset
        print("\hline \cline{1-8}", file=table_file)

        #We check if we need to print the average
        if average_list_print:
            list_averages = [elem for elem in self.data[:, -1]]
            print("\n\n", file=table_file)
            print('[', end='', file=table_file)
            for elem in list_averages[:-1]:
                print("%.4f" % elem, ", ", sep='', end='', file=table_file)
            print("%.4f" % list_averages[-1], "]", sep='', file=table_file)


        #We close the file:
        table_file.close()