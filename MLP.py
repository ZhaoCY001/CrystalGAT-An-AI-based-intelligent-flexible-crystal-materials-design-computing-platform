import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna(axis=0)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    if y.dtype.kind in {'U', 'S', 'O'}:
        le = LabelEncoder()
        y = le.fit_transform(y)
    return X, y

def train_model(model, X, y):
    X_train, X_unused, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    config = {
        "hidden_layer_sizes": (200, 100),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "batch_size": 64,
        "learning_rate_init": 5e-4,
        "max_iter": 1000,
        "random_state": 42
    }
    X, y = load_data("data.xlsx")
    model = MLPClassifier(hidden_layer_sizes=config["hidden_layer_sizes"],
                          activation=config["activation"],
                          solver=config["solver"],
                          alpha=config["alpha"],
                          batch_size=config["batch_size"],
                          learning_rate_init=config["learning_rate_init"],
                          max_iter=config["max_iter"],
                          random_state=config["random_state"])
    model = train_model(model, X, y)
    print("BP model training completed")
