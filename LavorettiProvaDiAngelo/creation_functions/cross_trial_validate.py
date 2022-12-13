from imblearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from dataset_creation import create_time_series_speed_up
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
import random
import numpy as np


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
