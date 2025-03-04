import pandas as pd
import os
import opendatasets as od
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Download the data
    dataset='https://www.kaggle.com/datasets/swadeshi/stress-detection-dataset/data'

    od.download(dataset,data_dir='./data/')
    df = pd.read_csv('./data/stress-detection-dataset/stress_detection.csv')
    
    #split data into train and test

    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    X_train.to_csv('./data/stress-detection-dataset/train.csv', index=False)
    X_test.to_csv('./data/stress-detection-dataset/test.csv', index=False)