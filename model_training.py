import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("HousingPrices-Amsterdam-August-2021.csv")
df = df.dropna()

centre_lat = 52.3731
centre_lon = 4.8926

dy = (df["Lat"] - centre_lat) * 111
dx = (df["Lon"] - centre_lon) * 68
df["Distance"] = np.sqrt(dx**2 + dy**2)

y = df.pop("Price")
X = df[["Area","Room","Lon","Lat","Distance"]]
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

print(X_train.describe().transpose())

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))
X_train_normalized = normalizer(np.array(X_train))

print(X_train_normalized[:5])
print(y_train[:5])
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

dnn_model.save("house_price_model.keras")

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
print(f"Test MAE: €{mae_metric.result().numpy():,.0f}")
