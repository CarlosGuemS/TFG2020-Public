import sys, pickle, datetime, os

##File extension
#Features + class
FILE_EXT_FEATURES = ".features"

def obtain_datetime_from_event(datetime_str:str):
    """
    Given the date and time from the sensor event in string format,
    the function builds and returns a datetime object

    :param date str: the date in format yyyy/mm/dd
    :param time str: the time in format hhhh:mm:ss.XXXXXX,
    where XXXXXX are microseconds
    :returns: a datetime object with the same values as the input
    :rtype: datetime
    """
    ###We drop the miliseconds since are measuring up to the second
    return datetime.datetime.strptime(datetime_str[:19], "%Y-%m-%d %H:%M:%S")

    year = int(date[0: 4])
    month = int(date[5: 7])
    day = int(date[8: 10])
    hours = int(time[0: 2])
    minutes = int(time[3: 5])
    seconds = int(time[6: 8])
    microseconds = int(time[9:] + "0"*(15-len(time)))

    return datetime.datetime(year, month, day, hours, minutes, seconds,
                             microseconds)

