import rampwf as rw

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, accuracy_score

problem_title = 'Stress estimation'

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

score_types = [
    rw.score_types.BalancedAccuracy(name='bal_acc', precision=3, adjusted=False),
    rw.score_types.Accuracy(name='acc', precision=3)
]
def _load_data(file, start=None, stop=None, load_waveform=True):
    if start is not None and stop is not None:
        nrows = stop - start
        X_df = pd.read_csv(file, skiprows=range(1, start + 1), nrows=nrows)
    else:
        X_df = pd.read_csv(file)

    y = X_df['Stress_Level_Biosensor']
    X_df = X_df.drop(columns=['Stress_Level_Biosensor', 'Stress_Level_Self_Report'], errors='ignore')

    # Replace None value in y by `-1
    y = y.fillna(-1).values

    return X_df, y

def get_train_data(path='.', start=None, stop=None, load_waveform=True):
    hash_train = hash((str(path), start, stop, load_waveform))
    if getattr(rw, "HASH_TRAIN", -1) == hash_train:
        return rw.X_TRAIN, rw.Y_TRAIN

    rw.HASH_TRAIN = hash_train

    train_file = Path(path) / 'student_health_data_preprocessed.csv'
    X_train, y_train = _load_data(train_file, start, stop, load_waveform)

    rw.X_TRAIN, rw.Y_TRAIN = X_train, y_train
    return X_train, y_train

def get_test_data(path='.', start=None, stop=None, load_waveform=True):
    hash_test = hash((str(path), start, stop, load_waveform))
    if getattr(rw, "HASH_TEST", -1) == hash_test:
        return rw.X_TRAIN, rw.Y_TRAIN

    rw.HASH_TEST = hash_test

    file = 'student_health_data_preprocessed.csv'
    file = Path(path) / file
    if os.environ.get("RAMP_TEST_MODE", False):
        start, stop = 0, 100
    rw.X_TEST, rw.Y_TEST = _load_data(file, start, stop, load_waveform)
    return rw.X_TEST, rw.Y_TEST

## Personalized Cross Validation
# to use it : problem.get_cv(X_train, y_train)
def get_cv(X, y):
    # Make sure the index is a range index so it is compatible with sklearn API
    X = X.reset_index(drop=True)

    chunks = X['chunk'].fillna('t')

    def split():
        train_idx = chunks[chunks != 'val'].index
        val_idx = chunks[chunks == 'val'].index
        yield train_idx, val_idx

    return split()