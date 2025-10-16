import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
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
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "gamma": 0.1,
        "eval_metric": "logloss",
        "random_state": 42,
        "use_label_encoder": False
    }
    X, y = load_data("data.xlsx")
    model = XGBClassifier(n_estimators=config["n_estimators"],
                          max_depth=config["max_depth"],
                          learning_rate=config["learning_rate"],
                          subsample=config["subsample"],
                          colsample_bytree=config["colsample_bytree"],
                          gamma=config["gamma"],
                          eval_metric=config["eval_metric"],
                          random_state=config["random_state"],
                          use_label_encoder=config["use_label_encoder"])
    model = train_model(model, X, y)
    print("XGBoost model training completed")