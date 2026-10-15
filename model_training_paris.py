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

X = df[["surface_reelle_bati", "nombre_pieces_principales",
        "latitude", "longitude"]]
y = df["valeur_fonciere"]

