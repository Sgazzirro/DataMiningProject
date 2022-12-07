import pandas as pd
import numpy as np


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


def create_time_series(dt_list, act_labels, trial_codes, actors, labeled=True):
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
    if labeled:
        dataset = np.zeros((0, (len(dt_list))+3))
    else:
        dataset = np.zeros((0, len(dt_list)))
    for subject in actors:
        for activity_code in act_labels:
            for trial_code in trial_codes[activity_code]:
                filename = '../LavorettiProvaDiMarco/A_DeviceMotion_data/'+activity_code+'_'+str(trial_code)+'/sub_'+str(int(subject))+'.csv'
                raw_data = pd.read_csv(filename)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                less_raw_data = np.zeros((len(raw_data), len(dt_list)))
                informations = []
                for index, sensor in enumerate(dt_list):
                    less_raw_data[:, index] = raw_data[sensor].values
                if labeled:
                    informations.append([subject, trial_code, activity_code])
                    informations = informations*len(less_raw_data)
                less_raw_data = np.concatenate((less_raw_data,informations), axis=1)
                dataset = np.append(dataset, less_raw_data, axis=0)

    information_columns = ["subject", "trial", "class"]
    total_columns = dt_list + information_columns
    print(total_columns)
    dataframe_format = pd.DataFrame(data=dataset, columns=total_columns)
    return dataframe_format

