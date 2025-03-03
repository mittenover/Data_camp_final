#%%
import pandas as pd
import numpy as np
import os

file_path = "./data/student_health_data.csv"
df = pd.read_csv(file_path)
print(df.head())
#%%
def categorize_stress(level):
    if 0 <= level < 2.5:
        return "Not Very Stressed"
    elif 2.5 <= level < 5:
        return "Stressed"
    elif 5 <= level < 7.5:
        return "Very Stressed"
    elif 7.5 <= level <= 10:
        return "Highly Stressed"
    else:
        return "Unknown"
# %%
# Appliquer la catégorisation
df["Stress_Category"] = df["Stress_Level_Biosensor"].apply(categorize_stress)

# Sauvegarder le fichier prétraité
df.to_csv("student_health_data_preprocessed.csv", index=False)
# %%
print(df.head())
# %%
