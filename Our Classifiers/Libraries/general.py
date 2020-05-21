import sys

#Datasets
sys.path.append('../Datasets')
import Kyoto1, Kyoto2, Kyoto3, Aruba

#WINDOW_SIZES = [5, 12, 19]
WINDOW_SIZES = [5, 12, 19, 26, 33]

##List of possible feature configurations
POSIBLE_FEATURE_CONFIG = ["BASE", "TD", "EMI", "TD+EMI"] #There's also ALL

def msg_error(msg:str):
    """
    Prints an error in the stderr output and ends the execution prematurely

    :param msg str: error message
    :rtype: None
    """
    print(msg, file=sys.stderr)
    sys.exit(1)

def load_dataset(argv: str):
    """
    Loads a module and returns it's placeholder name

    :param msg str: argument string
    :returns: the module and the palceholder name
    :rtype: module, str
    """
    dataset = sys.argv[3].upper()
    if dataset == "KYOTO1":
        return Kyoto1, "OA"
    elif dataset == "KYOTO2":
        return Kyoto2, "OAE"
    elif dataset == "KYOTO3":
        return Kyoto3, "IwA"
    elif dataset == "ARUBA":
        return Aruba, "DLR"
    else:
        #Unrecongnized dataset
        msg_error("Unrecognized dataset " + dataset)

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