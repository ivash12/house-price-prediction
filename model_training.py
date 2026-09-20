import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

df = pd.read_csv("HousingPrices-Amsterdam-August-2021.csv")
print(df.head())

y = df.pop("Price")
X = df[["Area","Room","Lon","Lat"]]
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

print(X_train.describe().transpose())

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)



