# From Subcription prediction to smart marketing

#Data Understanding & Audit

import pandas as pd

df = pd.read_csv("bank-full.csv", sep=";")

#print(df.head())
print("Dataset Dimensions:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nUnknown values in each column:")

for column in df.columns:
    unknown_count = (df[column] == "unknown").sum()

    if unknown_count > 0:
        print(column, ":", unknown_count)

print("\nTarget counts:")
print(df["y"].value_counts())

print("\nTarget percentages:")
print(df["y"].value_counts(normalize=True) * 100)

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

print("\nNumeric variable summary:")
print(df.describe())
