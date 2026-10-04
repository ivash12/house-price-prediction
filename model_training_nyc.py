import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("NY-House-Dataset.csv")
df = df.dropna()

y = df.pop("PRICE")

print(y.shape)