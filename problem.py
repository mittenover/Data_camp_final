import rampwf as rw

import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold


problem_title = 'stress estimation'

# Mapping int to categories
int_to_cat = {
   1 : 'NOT VERY STRESSED',
   2 : 'STRESSED',
   3 : 'VERY STRESSED',
   4 : 'HIGHLY STRESSED'
}

_event_label_int = list(int_to_cat)

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_event_label_int)

# An object implementing the workflow
workflow = rw.workflows.Classifier()

# Mapping categories to int
cat_to_int = {v: k for k, v in int_to_cat.items()}

# Define the score (specific to the competition)
score_types = [
    rw.score_types.BalancedAccuracy(name='bal_acc', precision=3, adjusted=False),
    rw.score_types.Accuracy(name='acc', precision=3)
]

def _get_data(path=".", split="train", cat_to_int = cat_to_int):
    # Load data from csv files into pd.DataFrame

    data_df = pd.read_csv(os.path.join(path, "data", split + ".csv"))

    data_df["cuisine1"] = data_df["cuisine1"].astype("category")
    data_df["cuisine2"] = data_df["cuisine2"].astype("category")

    # usefull columns
    subset = [
        "Student_ID",
        "Age",
        "Gender",
        "Heart_Rate",
        "Blood_Pressure_Systolic",
        "Blood_Pressure_Diastolic",
        # "Stress_Level_Biosensor",
        # "Stress_Level_Self_Report",
        "Physical_Activity",
        "Sleep_Quality",
        "Mood",
        "Study_Hours",
        "Project_Hours",
        "Health_Risk_Level"
    ]

    X = data_df[subset]

    # labels
    y = np.array(data_df["Stress_Level"].map(cat_to_int).fillna(-1).astype("int8"))

    return X, y

groups = None

def get_train_data(path="."):
    data = pd.read_csv(os.path.join(path, "data", "train.csv"))
    data["name"] = data["name"].astype("category")
    Name = np.array(data["name"].cat.codes)
    global groups
    groups = Name
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")

def get_cv(X, y):
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=2)
    return cv.split(X, y, groups)
