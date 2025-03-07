import pandas as pd
import os
import opendatasets as od

if __name__ == '__main__':
    # Download the data
    dataset='https://www.kaggle.com/datasets/swadeshi/stress-detection-dataset/data'

    od.download(dataset,data_dir='./data/')
    df = pd.read_csv('./data/stress-detection-dataset/stress_detection.csv')