from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
EPSILON = 1e-7  # same value Keras uses inside the Normalization layer


def load_weights(path):
    data = np.load(path)
    n_dense = int(data["n_dense"])
    return {
        "mean": data["mean"],
        "std": np.maximum(np.sqrt(data["variance"]), EPSILON),
        "layers": [(data[f"W{i}"], data[f"b{i}"]) for i in range(n_dense)],
    }


CITIES = {
    "amsterdam": {
        "weights": load_weights(BASE_DIR / "amsterdam.npz"),
        "features": ["area", "rooms", "lon", "lat", "distance"],
        "centre": (52.3731, 4.8926),
        "km_per_deg_lon": 68,
        "market_adjustment": 1.3,
        "bbox": (4.72, 52.28, 5.04, 52.43),  # west, south, east, north
    },
    "new-york": {
        "weights": load_weights(BASE_DIR / "new-york.npz"),
        "features": ["beds", "baths", "sqft", "lat", "lon", "distance"],
        "centre": (40.7580, -73.9855),
        "km_per_deg_lon": 84,
        "market_adjustment": 1.0,
        "bbox": (-74.27, 40.49, -73.68, 40.92),
    },
    "paris": {
        "weights": load_weights(BASE_DIR / "paris.npz"),
        "features": ["area", "rooms", "lat", "lon", "distance"],
        "centre": (48.8566, 2.3522),
        "km_per_deg_lon": 73,
        "market_adjustment": 1.0,
        "bbox": (2.224, 48.815, 2.470, 48.902),
    },
}

for name, cfg in CITIES.items():
    assert len(cfg["weights"]["mean"]) == len(cfg["features"]), f"{name}: feature count mismatch"


def run_model(weights, row):
    """Normalization -> Dense + ReLU -> Dense + ReLU -> Dense (same as the Keras model)."""
    x = (row - weights["mean"]) / weights["std"]
    last = len(weights["layers"]) - 1
    for i, (W, b) in enumerate(weights["layers"]):
        x = x @ W + b
        if i < last:
            x = np.maximum(x, 0)
    return x


def get_distance(city, lat, lon):
    centre_lat, centre_lon = CITIES[city]["centre"]
    dy = (lat - centre_lat) * 111
    dx = (lon - centre_lon) * CITIES[city]["km_per_deg_lon"]
    return np.sqrt(dx ** 2 + dy ** 2)


def is_inside_city(city, lat, lon):
    west, south, east, north = CITIES[city]["bbox"]
    return west <= lon <= east and south <= lat <= north


def predict_price(city, features):
    cfg = CITIES[city]
    features["distance"] = get_distance(city, features["lat"], features["lon"])
    row = np.array([[features[name] for name in cfg["features"]]])
    log_price = run_model(cfg["weights"], row)
    price = np.exp(log_price[0][0])
    return float(price) * cfg["market_adjustment"]
