import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("Paris-house-dataset.csv")
df = df.dropna(subset=["valeur_fonciere", "surface_reelle_bati",
            "nombre_pieces_principales", "latitude", "longitude"])
df = df.drop_duplicates()

print(df.shape)
df = df[(df["nature_mutation"] == "Vente") & (df["type_local"] == "Appartement")]
print(df.shape)
counts = df.groupby("id_mutation")["id_mutation"].transform("size")
df = df[counts == 1]
print(df.shape)

centre_lat = 48.8566
centre_lon = 2.3522

dy = (df["latitude"] - centre_lat) * 111
dx = (df["longitude"] - centre_lon) * 73
df["Distance"] = np.sqrt(dx**2 + dy**2)

X = df[["surface_reelle_bati", "nombre_pieces_principales",
        "latitude", "longitude", "Distance"]]
y = df["valeur_fonciere"]

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
    epochs=300,
    verbose=0,
    validation_split = 0.2)

dnn_model.save("house_price_model_paris.keras")

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

ae = np.abs(preds - y_test.values)
print("median abs error:", np.median(ae))
print("median % error:", np.median(ae / y_test.values))
print("baseline MAE:", np.mean(np.abs(y_test.values - y_train.median())))

