import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("HousingPrices-Amsterdam-August-2021.csv")
print(df.head())
df = df.dropna()
y = df.pop("Price")
X = df[["Area","Room","Lon","Lat"]]
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

print(X_train.describe().transpose())

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))
X_train_normalized = normalizer(np.array(X_train))

print(X_train_normalized[:5])
print(y_train[:5])
linear_model = tf.keras.Sequential([normalizer, tf.keras.layers.Dense(units=1)])

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mean_absolute_error')

history = linear_model.fit(
    X_train,
    np.log(y_train),
    epochs=100,
    verbose=0,
    validation_split = 0.2)

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

log_preds = linear_model.predict(X_test[:5])
preds = np.exp(log_preds).flatten()

for predicted, actual in zip(preds, y_test[:5]):
    print(f"Predicted: €{predicted:,.0f}   Actual: €{actual:,.0f}")



