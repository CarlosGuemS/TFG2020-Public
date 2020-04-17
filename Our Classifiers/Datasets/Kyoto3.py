import sys, os, pickle, datetime

sys.path.append("../Libraries")
import dataset_processing as dp

##PATH to data
#Data will be saved on the same spot
RAW_DATA_PATH = "../../CASAS/KYOTO3"
FORMATED_DATA_PATH = "../Datasets/KYOTO3"

##PARTICIPANTS
PARTICIPANTS = ["p04", "p13", "p14", "p15", "p17", "p18", "p20", "p23", "p24",
                "p25", "p26", "p27", "p28", "p29", "p30", "p31", "p33", "p34"]
                #p19 ignorado debido a errores en la entrada
                #p22 inexistente
                #26, 17, 31 modificados para rellenar la actividad
                #pero facil de asumir (!) cual era
RAW_FILE_EXTENSION = ".interwoven"

##ACTIVITIES
ACTIVITIY_NAMES = ["Fill medication dispenser", "Wash DVD", "Water plants",
                   "Answer the phone", "Prepare birthday card", "Prepare soup",
                   "Clean", "Choose outfit"]

##Sensor Data
NUM_EVENTS = 47
REAL_VALUE_EVENTS = set([41, 42, 43 ,45, 46])
"""
Sensor codes:
-[0-26] Motion Sensor MX 1-26,51
-[27-34] Item Sensor IX 1-8
-[35-40] Door Sensor DX 7-12
-[41-43] Water/Burner Sensor AD1-X A,B,C
-[44] Phone Sensor P01
-[45-46] Temperature Sensor TX 1-2
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
        if sensor_num == 50:
            #Special case, sensor M51:
            sensor_num = 26
    elif sensor[0] == "I": #IX
        if sensor == "I09": #Ignore I09
            return None, None
        sensor_num = (int(sensor[1:])) + 26
    elif sensor[0] == "D": #DX
        sensor_num = (int(sensor[1:])) + 28
    elif sensor == "P01": #P01
        sensor_num = 44
    elif sensor[0] == "A": #AD1
        sensor_num = 41 + _letter_to_num[sensor[-1]]
    elif sensor[0] == "T": #AD1
        sensor_num = (int(sensor[1:])) + 44
    elif sensor[0] == "E": #Unkown sensor E01
        return None, None
    
    #Unrecognized sensor: problem with the input data
    else:
        #Algo ha ocurrido
        print("Unkown sensor", sensor, file=sys.stderr)
        sys.exit(-1)
    
    #Obtain value number
    if sensor_num in REAL_VALUE_EVENTS:
        #With A sensors the value is already a number
        value_num = float(value)
    else:
        try:
            value_num = _message_to_value[value]
        except:
            print(sensor_num)
            print(sensor, value)
            sys.exit(0)

    return sensor_num, value_num

#Intermediate data generation
if __name__ == "__main__":
    #If the script runs, we will create archives with the intermediate data
    different_sensors = set()
    for participant in PARTICIPANTS:
        #We obtain the process events for all activities
        processed_events = []
        current_activity = 0

        #We build the file path
        complete_path = os.path.join(RAW_DATA_PATH,
                                     participant+RAW_FILE_EXTENSION)
        complete_path = os.path.normpath(complete_path)
        #We analize the file 
        input_file = open(complete_path, 'r')
        input_data = input_file.read().split("\n")
        for event in map(lambda x: x.split(), input_data):

            if not len(event):
                continue

            timestamp = dp.obtain_datetime_from_event(event[0], event[1])
            different_sensors.add(event[2])
            sensor, value = obtain_num_from_sensor(event[2], event[3])
            if sensor is None or value is None:
                #Invalid input
                continue
            activity = int(event[4]) - 1

            #We store the result
            processed_events.append([timestamp, sensor, value, activity])

        #Once we obtained all process events we save it on a file
        new_file_path = os.path.join(FORMATED_DATA_PATH,
                                     participant + dp.FILE_EXT_FEATURES)
        new_file_path = os.path.normpath(new_file_path)
        new_file = open(new_file_path, "wb")
        pickle.dump(processed_events, new_file)
        new_file.close()
    
    print(sorted(list(different_sensors)))

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
