import pandas as pd
import numpy as np
from feature_extraction import extract_features

def select_sensors(sensors):
    """
    Select the sensors to shape the final dataset.

    Args:
    sensors: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration]

    Returns:
    It returns a list of columns to use for creating time-series from files.
    """
    columns = []
    for sensor in sensors:
        columns.append(sensor + ".x")
        columns.append(sensor + ".y")
        columns.append(sensor + ".z")
    return columns


def create_time_series_speed_up(dt_list, act_labels, trial_codes, actors, labeled=True):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activities
        trial_codes: dictionary with shape <Code_Activity> -> <List of Trials we are interested in>
        actors: codes of subjects which we want to include in the dataset
        labeled: True, if we want a labeled dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.

    """
    dataset = pd.DataFrame()
    information_columns = ["subject", "trial", "class"]
    for subject in actors:
        for activity_code in act_labels:
            for trial_code in trial_codes[activity_code]:
                filename = '../LavorettiProvaDiMarco/A_DeviceMotion_data/'+activity_code+'_'+str(trial_code)+'/sub_'+str(int(subject))+'.csv'
                raw_data = pd.read_csv(filename)
                raw_data = raw_data.drop(['Unnamed: 0', "attitude.pitch", "attitude.roll", "attitude.yaw", "gravity.x",
                                          "gravity.y", "gravity.z"], axis=1)
                data_collapsed = extract_features(raw_data, activity_code, 150)
                informations = []
                if labeled:
                    data_collapsed["class"] = activity_code
                dataset = pd.concat((dataset, data_collapsed), axis=0)

    dataframe_format = pd.DataFrame(data=dataset)
    return dataframe_format
