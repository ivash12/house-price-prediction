from pathlib import Path
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).parent / "house_price_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

centre_lat = 52.3731
centre_lon = 4.8926

def get_distance(lat, lon):
    dy = (lat - centre_lat) * 111
    dx = (lon - centre_lon) * 68
    return np.sqrt(dx ** 2 + dy ** 2)


def predict_price(area, rooms, lon, lat):
    distance = get_distance(lat, lon)
    row = np.array([[area, rooms, lon, lat, distance]])
    log_price = model.predict(row)
    price = np.exp(log_price[0][0])
    return float(price)
