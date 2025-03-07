import rampwf as rw
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Stress_estimation'

# Mapping int to categories
int_to_cat = {
   0 : 'Low Stress',
   1 : 'Moderate Stress',
   2 : 'High Perceived Stress'
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

    y = X_df['Stress_Category'].map(cat_to_int)    # Convert categories to int
    X_df = X_df.drop(columns=['Stress_Category', 'PSS_score'], errors='ignore') # Drop the target columns from the features

    # Replace None value in y by `-1
    y = y.fillna(-1).values

    return X_df, y

def get_split(path='.'):
    file = Path(path) / 'data/stress-detection-dataset/preprocessed_stress_detection.csv'
    X, y = _load_data(file)
    lines = len(X)
    start_first_lines = 0
    end_first_lines = int(lines * 0.8)-1
    start_other_lines = end_first_lines
    end_other_lines = lines

    return start_first_lines, end_first_lines, start_other_lines, end_other_lines

def get_train_data(path='.'):
    hash_train = hash((str(path)))
    if getattr(rw, "HASH_TRAIN", -1) == hash_train:
        return rw.X_TRAIN, rw.Y_TRAIN

    rw.HASH_TRAIN = hash_train

    train_file = Path(path) / 'data/stress-detection-dataset/preprocessed_stress_detection.csv'

    start_first_lines, end_first_lines, start_other_lines, end_other_lines = get_split(path)
    X_train, y_train = _load_data(train_file, start=start_first_lines, stop=end_first_lines)

    rw.X_TRAIN, rw.Y_TRAIN = X_train, y_train
    return X_train.to_numpy(), y_train

def get_test_data(path='.'):
    hash_test = hash((str(path)))
    if getattr(rw, "HASH_TEST", -1) == hash_test:
        return rw.X_TRAIN, rw.Y_TRAIN

    rw.HASH_TEST = hash_test

    file = 'data/stress-detection-dataset/preprocessed_stress_detection.csv'
    file = Path(path) / file

    start_first_lines, end_first_lines, start_other_lines, end_other_lines = get_split(path)
    
    if os.environ.get("RAMP_TEST_MODE", False):
        start, stop = 0, 100
    else:
        start, stop = start_other_lines, end_other_lines
    rw.X_TEST, rw.Y_TEST = _load_data(file, start=start, stop=stop)
    return rw.X_TEST.to_numpy(), rw.Y_TEST

## Personalized Cross Validation
# to use it : problem.get_cv(X_train, y_train)
def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=57)
    return cv.split(X, y)