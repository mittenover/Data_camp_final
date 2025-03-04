import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def categorize_stress(level):
    if 0 <= level < 14:
        return "Low Stress"
    elif 14 <= level < 27:
        return "Moderate Stress"
    elif 27 <= level < 40:
        return "High Perceived Stress"
    else:
        return "Unknown"
    
if __name__ == '__main__':

    file_path = "./data/stress-detection-dataset/stress_detection.csv"
    df = pd.read_csv(file_path)
    df["Stress_Category"] = df["PSS_score"].apply(categorize_stress)
    df.to_csv("./data/stress-detection-dataset/preprocessed_stress_detection.csv", index=False)
    df1 = pd.read_csv("./data/stress-detection-dataset/preprocessed_stress_detection.csv")
    train, test = train_test_split(df1, test_size=0.2, random_state=42)
    train.to_csv("./data/stress-detection-dataset/train.csv", index=False)
    test.to_csv("./data/stress-detection-dataset/test.csv", index=False)
    print("Preprocessing done.")
