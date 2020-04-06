import sys, os, pickle, datetime

sys.path.append("../Libraries")
import dataset_processing as dp

##PATH to data
#Data will be saved on the same spot
RAW_DATA_PATH = "../../CASAS/KYOTO2"
FORMATED_DATA_PATH = "../Datasets/KYOTO2"

##PARTICIPANTS
PARTICIPANTS = ["p17", "p18", "p20", "p21", "p22", "p23", "p24","p26", "p27",
                "p29", "p30", "p31", "p52", "p53", "p54", "p55", "p56", "p57",
                "p58", "p59"]
##ACTIVITIES
"""
t1: Make a phone call
t2: Wash hands
t3: Cook
t4: Eat
t5: Clean
"""
ACTIVITIES = [".t1", ".t2", ".t3", ".t4", ".t5"]
ACTIVITIY_NAMES = ["Make a phone call", "Wash hands", "Cook", "Eat",
                   "Clean"]

##Sensor Data
NUM_EVENTS = 39
REAL_VALUE_EVENTS = set([35,36,37])
"""
Sensor codes:
-[0-25] Motion Sensor MX 1-26
-[26-33] Item Sensor IX 1-8
-[34] Door Sensor D01
-[35-37] Water/Burner Sensor AD1-X A,B,C
-[38] Phone Sensor
"""
##Obtaining numeric values from sensor data
_letter_to_num = {"A":0, "B":1, "C":2}
_message_to_value = {"OPEN":1.0, "ABSENT":1.0, "ON":1.0, "START":1.0,
                    "END":0.0, "CLOSE":0.0, "OFF":0.0, "PRESENT":0.0}
#Unknown sensor "E01"
_message_to_value["STOP_INSTRUCT"] = 0
_message_to_value["START_INSTRUCT"] = 1
def obtain_num_from_sensor(sensor: str, value: str):
    """
    Given a sensor and its vale obtain its numeric representation.

    :param sensor str: the sensor code
    :param value str: the value of the given sensor
    :returns: a tuple, with the numeric value of the sensor and its value
    :rtype: (int, float)
    """
    #Obtain sensor number
    if sensor[0] == "M":  #MX
        sensor_num = (int(sensor[1:])) - 1
    elif sensor[0] == "I": #IX
        sensor_num = (int(sensor[1:])) + 25
    elif sensor == "D01": #D01
        sensor_num = 34
    elif sensor == "asterisk": #Telephone
        sensor_num = 38
    elif sensor[0] == "A": #AD1
        sensor_num = 35 + _letter_to_num[sensor[-1]]
    #Unkown sensor E01
    elif sensor[0] == "E": #E01
        return None, None
    #Unrecognized sensor: problem with the input data
    else:
        #Algo ha ocurrido
        print("Unkown sensor", sensor, file=sys.stderr)
        sys.exit(-1)
    
    #Obtain value number
    if sensor[0] == "A":
        #With A sensors the value is already a number
        value_num = float(value)
    else:
        value_num = _message_to_value[value]

    return sensor_num, value_num

#Intermediate data generation
if __name__ == "__main__":
    #Generation of the data
    def _generate_intermediate_data(input_file_path:str,
                                    obtain_num_from_sensor):
        """
        Creates a python vector with the data contained in the fiven file

        :param input_file_path str: path of the file to analyze
        :param obtain_num_from_sensor function: the python function
        responsible of obtaining the numeric representation of the sensor and
        data
        :returns: a generator of the file
        :rtype: gen-object
        """

        input_file = open(input_file_path, 'r')
        activity = int(input_file_path[-1])
        #Reads all the lines in the file
        for sensor_event in input_file.read().split('\n'):
            if len(sensor_event) == 0:
                continue
            #Obtain the different parts of the sensor event
            divided_sensor_event = sensor_event.split('\t')
            timestamp = dp.obtain_datetime_from_event(divided_sensor_event[0],
                                                              divided_sensor_event[1])
            sensor, value = obtain_num_from_sensor(divided_sensor_event[2],
                                        divided_sensor_event[3])
            
            #If there wasn't any problem with the sensor, it gets rebuilt
            if not (sensor is None or value is None):
                yield [timestamp, sensor, value, activity]
        #We close the file
        input_file.close()

    #If the script runs, we will create archives with the intermediate data
    for participant in PARTICIPANTS:
        #We obtain the process events for all activities
        processed_events = []
        for activity in ACTIVITIES:
            #We build the file path
            complete_path = os.path.join(RAW_DATA_PATH,
                                         participant+activity)
            complete_path = os.path.normpath(complete_path)
            #We feed the file path to the 
            event_generator = _generate_intermediate_data(complete_path,
                                                                     obtain_num_from_sensor)
            processed_events += list(event_generator)
        #Once we obtained all process events we save it on a file
        new_file_path = os.path.join(FORMATED_DATA_PATH,
                                     participant + dp.FILE_EXT_FEATURES)
        new_file_path = os.path.normpath(new_file_path)
        new_file = open(new_file_path, "wb")
        pickle.dump(processed_events, new_file)
        new_file.close()

#Accesing the intermediate data
def obtaining_data(fold:int = 1, *args):
    """
    Obtains and aggregates the data obtained from the Kyoto dataset.
    
    :param fold int: modify the fold
    :param args list: other parameters to be ignored
    :returns: the list of training data and of test data
    :rtype: list, list
    """

    #Divide the participants
    try:
        training_participants = PARTICIPANTS[:-fold]
        testing_participants = PARTICIPANTS[-fold:]
    except:
        raise Exception("Invalid fold for the dataset")
    
    #Obtaining training data
    aggregated_training_data = []
    for participant in training_participants:
        complete_path = os.path.join(FORMATED_DATA_PATH,
                                     participant+dp.FILE_EXT_FEATURES)
        complete_path = os.path.normpath(complete_path)
        temp_file = open(complete_path, 'rb')
        temp_data = pickle.load(temp_file)
        temp_file.close()
        aggregated_training_data.append(temp_data)

    #Obtaining test data
    aggregated_test_data = []
    for participant in testing_participants:
        complete_path = os.path.join(FORMATED_DATA_PATH,
                                     participant+dp.FILE_EXT_FEATURES)
        complete_path = os.path.normpath(complete_path)
        temp_file = open(complete_path, 'rb')
        temp_data = pickle.load(temp_file)
        temp_file.close()
        aggregated_test_data.append(temp_data)
    
    return aggregated_training_data, aggregated_test_data
