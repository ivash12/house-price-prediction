import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("NY-House-Dataset.csv")
df = df.dropna()
df = df.drop_duplicates()

# df = df[
#     (df["LATITUDE"] >= 40.49) & (df["LATITUDE"] <= 40.92) &
#     (df["LONGITUDE"] >= -74.27) & (df["LONGITUDE"] <= -73.68)
# ]

y = df.pop("PRICE")
X = df[["BEDS", "BATH", "PROPERTYSQFT", "LATITUDE", "LONGITUDE"]]

print(X.head())
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
print(X_train.describe().transpose())