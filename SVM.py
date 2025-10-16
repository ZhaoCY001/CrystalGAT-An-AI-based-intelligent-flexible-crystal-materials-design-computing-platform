import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna(axis=0)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    if y.dtype.kind in {'U', 'S', 'O'}:
        le = LabelEncoder()
        y = le.fit_transform(y)
    X, y = shuffle(X, y, random_state=42)
    return X, y

def train_model(model, X, y):
    X_train, X_unused, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    config = {
        "kernel": "rbf",
        "C": 0.5,
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42
    }
    X, y = load_data("data.xlsx")
    model = SVC(kernel=config["kernel"],
                C=config["C"],
                gamma=config["gamma"],
                class_weight=config["class_weight"],
                probability=config["probability"],
                random_state=config["random_state"])
    model = train_model(model, X, y)
    print("SVM model training completed")
