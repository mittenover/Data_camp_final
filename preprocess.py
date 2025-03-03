#%%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

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
# Vérifier les valeurs manquantes
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)
# %%
# Charger le fichier prétraité
df = pd.read_csv("student_health_data_preprocessed.csv")

# Diviser les données en ensembles d'entraînement et de test
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Stress_Level_Biosensor"])

# Sauvegarder les ensembles d'entraînement et de test
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
# %%
