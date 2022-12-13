import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
import random
from scipy.stats import kurtosis, skew
from numpy.fft import fft
import detecta
from scipy.signal import savgol_filter


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


def create_time_series_speed_up(act_labels, trial_codes, actors, labeled=True, mode="Collapsed", exclude_5=False):
    """
    Args:
        act_labels: list of activities
        trial_codes: dictionary with shape <Code_Activity> -> <List of Trials we are interested in>
        actors: codes of subjects which we want to include in the dataset
        labeled: True, if we want a labeled dataset. False, if we only want sensor values.
        mode:
        exclude_5:
    Returns:
        It returns a time-series of sensor data.

    """
    dataset = pd.DataFrame()
    information_columns = ["subject", "trial", "class"]
    for subject in actors:
        for activity_code in act_labels:
            for trial_code in trial_codes[activity_code]:
                filename = '../LavorettiProvaDiMarco/A_DeviceMotion_data/'+activity_code+'_'+str(trial_code)+'/sub_'+str(int(subject))+'.csv'
                if exclude_5:
                    if subject == 5 and activity_code == "sit" and trial_code == 13:
                        subject = 4
                raw_data = pd.read_csv(filename)
                raw_data = raw_data.drop(['Unnamed: 0', "attitude.pitch", "attitude.roll", "attitude.yaw", "gravity.x",
                                          "gravity.y", "gravity.z"], axis=1)
                if mode == "raw":
                    data_collapsed = raw_data
                else:
                    data_collapsed = extract_features(raw_data, activity_code, 150)
                if labeled:
                    data_collapsed["class"] = activity_code
                dataset = pd.concat((dataset, data_collapsed), axis=0)

    dataframe_format = pd.DataFrame(data=dataset)
    return dataframe_format



def get_a_split():

    # 1. Randomly choose 20 over 24 subjects
    train_subjects, test_subjects = train_test_split(np.array(range(1, 25)), test_size=(1/6),
                                                     random_state=15, shuffle=True)
    # 2. Load all experiments for train_subjects
    ACTIVITY_CODES = ["dws", "jog", "sit", "std", "ups", "wlk"]

    TRIAL_CODES = {
        ACTIVITY_CODES[0]:[1,2,11],
        ACTIVITY_CODES[1]:[9,16],
        ACTIVITY_CODES[2]:[5,13],
        ACTIVITY_CODES[3]:[6,14],
        ACTIVITY_CODES[4]:[3,4,12],
        ACTIVITY_CODES[5]:[7,8,15]
    }
    X_train = create_time_series_speed_up(ACTIVITY_CODES, TRIAL_CODES, train_subjects, True)
    X_test = create_time_series_speed_up(ACTIVITY_CODES, TRIAL_CODES, test_subjects, True)

    y_test = X_test["class"]
    X_test = X_test.drop(["class"], axis=1)

    y_train = X_train["class"]
    X_train = X_train.drop(["class"], axis=1)

    return X_train, X_test, y_train, y_test


def tune_parameters(X_train, y_train, estimators):
    selector = SelectKBest(k=8)
    classifier = RandomForestClassifier(n_estimators=estimators)
    pipeline = make_pipeline(selector, classifier)

    output = cross_validate(pipeline,
                            X_train,
                            y_train,
                            scoring={
                                "accuracy" : make_scorer(accuracy_score),
                                "f1_score" : make_scorer(f1_score, average="weighted"),
                                "recall" : make_scorer(recall_score, average="weighted"),
                                "precision" : make_scorer(precision_score, average="weighted")
                            },
                            return_estimator=True,
                            cv=StratifiedKFold(n_splits=10))
    return output


def noise_filter(dataframe):
    for column in dataframe.columns:
        dataframe[column] = savgol_filter(dataframe[column], 5, 2, axis=0)
    return dataframe


def compute_peaks(data):
    X=fft(data)
    N=len(X)
    n=np.arange(N)
    sr=1/50
    T=N/sr
    freq=n/T
    c=20

    n_oneside=N//2
    x=freq[1:n_oneside]
    y=np.abs(X[1:n_oneside])

    max_peak_height=np.amax(y)/c
    peaks=[]
    while len(peaks)<5:
        peaks=detecta.detect_peaks(y, mph=max_peak_height)
        c+=5
        max_peak_height=np.amax(y)/c
    peaks_x=peaks/T
    peaks_y=y[peaks]

    return peaks_x[0:5], peaks_y[0:5]


def find_fft_points(data, name):
    (indices_peaks, peaks) = compute_peaks(data)
    columns_x=[name + "X#1", name + "X#2", name + "X#3", name + "X#4", name + "X#5"]
    columns_y=[name + "P#1", name + "P#2", name + "P#3", name + "P#4", name + "P#5"]
    x_p = pd.DataFrame(data=indices_peaks).T
    x_p.columns = columns_x
    y_p = pd.DataFrame(data=peaks).T
    y_p.columns = columns_y
    tot_p = pd.concat([x_p, y_p], axis=1)
    return tot_p


def compute_time_features(df):
    data_total = pd.DataFrame()
    for column in df.columns:
        temp = find_time_features(df[column], column)
        data_total = pd.concat([data_total, temp], axis=1)
    return data_total


def find_time_features(data, name):
    columns = [name + "_mean", name + "_std", name + "_range", name + "_IRQ", name + "_kurtosis", name + "_skewness"]
    properties = [np.mean(data), np.std(data), np.max(data) - np.min(data),
                  np.quantile(data, 0.75) - np.quantile(data, 0.25), kurtosis(data), skew(data)]
    d = pd.DataFrame(data=properties).T
    d.columns = columns
    return d


def compute_freq_features(df):
    data_total = pd.DataFrame()
    for column in df.columns:
        temp = find_fft_points(df[column], column)
        data_total = pd.concat([data_total, temp], axis=1)
    return data_total


def collapse(df):
    time_df=compute_time_features(df)
    freq_df=compute_freq_features(df)
    return pd.concat([time_df, freq_df], axis=1)


def extract_features(dataframe, class_name, sample_number):
    i = dataframe.shape[0]//sample_number
    j=0
    filtered_df=noise_filter(dataframe)
    df_time_series=pd.DataFrame()
    for count in range(1,i):
        samples_df=filtered_df.iloc[j:sample_number*count, :]
        new_df=collapse(samples_df)
        if(count==1):
            df_time_series=new_df
        else:
            df_time_series = pd.concat([df_time_series, new_df], axis=0)
        j=sample_number*count
    return df_time_series

