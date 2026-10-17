"""
Export the trained Keras models to small .npz files that the website can use
without TensorFlow.

Run from the project root after (re)training a model:
    python export_weights.py

For every model it saves the normalizer's mean/variance and each Dense layer's
weights and bias, then checks that the NumPy forward pass gives the same
predictions as TensorFlow.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

PREDICTOR_DIR = Path(__file__).parent / "housesite" / "predictor"

MODELS = {
    "amsterdam": "house_price_model.keras",
    "new-york": "house_price_model_nyc.keras",
    "paris": "house_price_model_paris.keras",
}

EPSILON = tf.keras.backend.epsilon()


def numpy_forward(weights, x):
    """Same calculation as the Keras model, written with NumPy."""
    std = np.maximum(np.sqrt(weights["variance"]), EPSILON)
    x = (x - weights["mean"]) / std
    n_dense = int(weights["n_dense"])
    for i in range(n_dense):
        x = x @ weights[f"W{i}"] + weights[f"b{i}"]
        if i < n_dense - 1:
            x = np.maximum(x, 0)  # ReLU on every layer except the last one
    return x


for city, filename in MODELS.items():
    model = tf.keras.models.load_model(PREDICTOR_DIR / filename)

    norm = next(l for l in model.layers if isinstance(l, tf.keras.layers.Normalization))
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]

    weights = {
        "mean": np.array(norm.mean).reshape(-1),
        "variance": np.array(norm.variance).reshape(-1),
        "n_dense": np.array(len(dense_layers)),
    }
    for i, layer in enumerate(dense_layers):
        W, b = layer.get_weights()
        weights[f"W{i}"] = W
        weights[f"b{i}"] = b

    # check: random inputs on a realistic scale, compare TF and NumPy outputs
    rng = np.random.default_rng(0)
    x = weights["mean"] + np.sqrt(weights["variance"]) * rng.standard_normal((200, len(weights["mean"])))
    x = x.astype("float32")
    tf_out = model.predict(x, verbose=0)
    np_out = numpy_forward(weights, x)
    max_diff = float(np.max(np.abs(tf_out - np_out)))

    out_path = PREDICTOR_DIR / f"{city}.npz"
    np.savez(out_path, **weights)
    status = "OK" if max_diff < 1e-4 else "MISMATCH - do not deploy"
    print(f"{city:10s} -> {out_path.name}  inputs={len(weights['mean'])}  max diff={max_diff:.2e}  {status}")
