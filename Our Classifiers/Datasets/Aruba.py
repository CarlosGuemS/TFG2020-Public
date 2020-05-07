import sys, os, pickle, datetime
from operator import itemgetter

sys.path.append("../Libraries")
import dataset_processing as dp

##PATH to data
#Data will be saved on the same spot
RAW_DATA_PATH = "../../CASAS/ARUBA"
FORMATED_DATA_PATH = "../Datasets/ARUBA"
FILE_NAME = "aruba"

##ACTIVITIES
ACTIVITIY_NAMES = ["Meal_Preparation", "Relax", "Eating", "Work", "Sleeping",
                   "Wash_Dishes", "Bed_to_Toilet", "Enter_Home", "Leave_Home",
                   "Housekeeping", "Respirate"]

##Sensor Data
NUM_EVENTS = 42
REAL_VALUE_EVENTS = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
"""
Sensor codes:
-[0-30] Motion Sensor MX 001-031
-[31-34] Door Sensor DX 0001-0004
-[35-39] Temperature Sensor TX 001-005
-[40] ENTERHOME
-[41] LEAVEHOME
"""
_message_to_value = {"ON":1.0, "OPEN":1.0, "OFF":0.0, "CLOSE":0.0}
def obtain_num_from_sensor(sensor: str, value: str):
    """
    Given a sensor and its vale obtain its numeric representation.

    :param sensor str: the sensor code
    :param value str: the value of the given sensor
    :returns: a tuple, with the numeric value of the sensor and its value
    :rtype: (int, float)
    """
    #Trivial cases ENTERHOME and LEAVEHOME
    if sensor == "ENTERHOME":
        return 40, float(value)
    elif sensor == "LEAVEHOME":
        return 41, float(value)

    #Obtain sensor number
    if sensor[0] == "M":  #Motion
        sensor_num = 0
    elif sensor[0] == "D": #Door
        sensor_num = 31
    elif sensor[0] == "T": #Temperature
        sensor_num = 35
    else:
        print("Unkown sensor", sensor, file=sys.stderr)
        sys.exit(-1)
    sensor_num += int(sensor[1:])-1
    
    #Obtain value number
    if sensor[0] == "T":
        #With A sensors the value is already a number
        value_num = float(value)
    else:
        value_num = _message_to_value[value]

    return sensor_num, value_num

#Generating the intermediate data
if __name__ == "__main__":

    current_activity = -1
    activity_ends = False

    #If the script runs, we will create one file with all the intermediate data
    input_file_path = os.path.join(RAW_DATA_PATH,"data")
    input_file = open(input_file_path, 'r')
    input_data = input_file.read().split("\n")
    formatted_data = []
    #We build the features
    for event in map(lambda x: x.split(), input_data):
        if not len(event):
            continue
        
        #First we check the activity it belongs
        activity_ends = False
        try:
            if len(event) == 6:
                if event[-1] == "begin":
                    current_activity = ACTIVITIY_NAMES.index(event[-2])
                elif event[-1] == "end":
                    activity_ends = True
                else:
                    raise Exception()
        except:
            print("Unknown class", event, file=sys.stderr)
            sys.exit(-1)
        
        #If the event doesn't belong to an activity, we discard it
        if not current_activity == -1:
            #Obtain the formated event
            timestamp = dp.obtain_datetime_from_event(event[0] + " " + event[1])
            sensor, value = obtain_num_from_sensor(event[2], event[3])
            #If there wasn't any problem with the sensor, it gets rebuilt
            if not (sensor is None or value is None):
                formatted_data.append([timestamp, sensor, value,
                                    current_activity])
        
        #We check if the activity ended
        if activity_ends:
            current_activity = -1
        
    #We need to order formatted data:
    formatted_data.sort(key=itemgetter(0))

    #Once we obtained all process events we save it on a file
    new_file_path = os.path.join(FORMATED_DATA_PATH,
                                 FILE_NAME + dp.FILE_EXT_FEATURES)
    new_file_path = os.path.normpath(new_file_path)
    new_file = open(new_file_path, "wb")
    pickle.dump(formatted_data, new_file)
    new_file.close()

    print(formatted_data[:10])

#Accesing the intermediate data
def invalid_date(month, day):
    print("Invalid date", "month =", month, "day =", day, file=sys.stderr)
    sys.exit(-1)
def obtaining_data(month:int = 5, day:int = 7, *args):
    """
    Obtains and aggregates the data obtained from the ARUBA dataset. The test
    data starts at the date indicated
    
    :param month int: month of the date indicating where the test data starts
    :param day int: day of the date indicating where the test data starts
    :param args list: other parameters to be ignored
    :returns: the list of training data and of test data
    :rtype: list, list
    """

    #We check the given date is correct
    if month == 11:
        if day not in range(4,31):
            invalid_date(month, day)
    elif month == 6:
        if day not in range(1, 12):
            invalid_date(month, day)
    elif month in [12, 1, 3, 5]:
        if day not in range(1, 32):
            invalid_date(month, day)
    elif month == 4:
        if day not in range(1, 31):
            invalid_date(month, day)
    elif month == 2:
        if day not in range(1, 29):
            invalid_date(month, day)
    else:
        invalid_date(month, day)


    #We don't divide the data here, we send all as training data
    complete_path = os.path.join(FORMATED_DATA_PATH,
                                 FILE_NAME +dp.FILE_EXT_FEATURES)
    complete_path = os.path.normpath(complete_path)
    temp_file = open(complete_path, 'rb')
    temp_data = pickle.load(temp_file)

    #We prepare both datasets
    training_data = []; test_data = []
    current_data = training_data; training = True

    #We split the data
    for event in temp_data:
        #We check the date to see if we have to change
        if training:
            date = event[0]
            if date.month == month and date.day == day:
                training = False
                current_data = test_data
        
        current_data.append(event)


    return [training_data], [test_data]
