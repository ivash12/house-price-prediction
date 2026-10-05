import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("NY-House-Dataset.csv")
df = df.dropna()
df = df.drop_duplicates()

centre_lat = 40.7580
centre_lon = -73.9855

dy = (df["LATITUDE"] - centre_lat) * 111
dx = (df["LONGITUDE"] - centre_lon) * 84
df["Distance"] = np.sqrt(dx**2 + dy**2)

y = df.pop("PRICE")
X = df[["BEDS", "BATH", "PROPERTYSQFT", "LATITUDE", "LONGITUDE", "Distance"]]

print(X.head())
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
print(X_train.describe().transpose())

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))
X_train_normalized = normalizer(np.array(X_train))
print(X_train_normalized[:5])

dnn_model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

dnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error')

history = dnn_model.fit(
    X_train,
    np.log(y_train),
    epochs=500,
    verbose=0,
    validation_split = 0.2)

dnn_model.save("house_price_model_nyc.keras")

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

log_preds = dnn_model.predict(X_test)
preds = np.exp(log_preds).flatten()


mae_metric = tf.keras.metrics.MeanAbsoluteError()
mae_metric.update_state(y_test.values, preds)
print(f"Test MAE: ${mae_metric.result().numpy():,.0f}")

