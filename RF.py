import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
        "n_estimators": 300,
        "max_depth": 4,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1
    }
    X, y = load_data("data.xlsx")
    model = RandomForestClassifier(n_estimators=config["n_estimators"],
                                   max_depth=config["max_depth"],
                                   min_samples_split=config["min_samples_split"],
                                   min_samples_leaf=config["min_samples_leaf"],
                                   class_weight=config["class_weight"],
                                   random_state=config["random_state"],
                                   n_jobs=config["n_jobs"])
    model = train_model(model, X, y)
    print("Random Forest model training completed")