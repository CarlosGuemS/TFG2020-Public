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
        print(*self.data, sep=",", file = table_file)
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
    
    def add_confusion_matrix(self, header:str, confusion_matrix):
        """
        Stores a given confusion matrix in the file

        :param header str: header for the confusion matrix
        :param confusion_matrix: a, not normalized, confusion matrix
        """
        #Print the header
        print(header, file=self.file)
        
        #Obtain the averages (normalizes it)
        for ii in range(self.number_activities):
            total = np.sum(confusion_matrix[ii, :])
            total = total if total else 1
            confusion_matrix[ii, :] = confusion_matrix[ii,:] / total
        
        #Print the matrix
        print(confusion_matrix, end = "\n\n\n", file = self.file)

#Latex accuracy table
class Latex_Table:
    """
    Creates a table to store the results to later be printed as the contents
    of a table in LaTex.

    :param entries_per_row list: number of entries per row
    :param dataset_name str: name of the dataset
    """

    #We lay out the constants: headers and similar
    row_headers = ["\multirow{9}{*}{\PLACEHOLDER} & Base",
                  "\cline{2-8} & Base + PWA", "\cline{2-8} & Base + TD",
                  "\cline{2-8} & Base + EMI", "\cline{2-8} & Base + PWA + TD",
                  "\cline{2-8} & Base + PWA + EMI", "\cline{2-8} & Base + TD + EMI",
                  "\cline{2-8} & Base + PWA + TD"]
    
    intermediate_rows = ("& + EMI  &  &  &  &  &  & \\\\",)

    ind_rows_first_part = (0, 7)
    ind_rows_second_part = (4, 8)
    num_rows = 8

    def __init__(self, entries_per_row:int, dataset_name:str):
        #Array to store the data
        self.data = np.zeros(shape=(self.num_rows, entries_per_row))
        #Indices (to know how to store the array)
        self.index = 0
        #Number of entries per row
        self.entries_per_row = entries_per_row
        #We change the dataset name
        self.row_headers[0] = self.row_headers[0].replace("PLACEHOLDER", dataset_name, 1)

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
        :param num_decimal int: number of decimals to print
        :param average_list_print: indicates if the total averages are printed
        as a separate list. Used to generate graphs in matplotlib
        """
        #We ouput an error if the file isn't full yet
        if self.index != self.num_rows:
            msg_error("Error: Table of results not full")
        
        #We open the file:
        table_file = open(file_name + ".table", 'w')
        
        #We print the first set of rows
        for ii in range(*self.ind_rows_first_part):
            #Row header
            print(self.row_headers[ii], end='', file=table_file)
            #Row content
            for elem in self.data[ii, :]:
                val_to_print = _round_decimal(elem, num_decimal)
                print(" & ", val_to_print, sep='', end='', file=table_file)
            #Row end
            print(' \\\\', file=table_file)
        
        #We print the last row
        #Row header
        print(self.row_headers[7], end='', file=table_file)
        #Row content
        for elem in self.data[7, :]:
            val_to_print = _round_decimal(elem, num_decimal)
            print(" & \multirow{2}{*}{", val_to_print, "}", sep='',
                    end='', file=table_file)
        #Row end
        print(' \\\\', file=table_file)
        #Print intermediate line
        print(self.intermediate_rows[0], file=table_file)
    
        
        #We print the horizontal line diving this dataset
        print("\hline \cline{1-8}", file=table_file)


        if average_list_print:
            print("\n\n", file=table_file)
            print([float(_round_decimal(elem, num_decimal))
                   for elem in self.data[:, -1]],
                  file=table_file)


        #We close the file:
        table_file.close()