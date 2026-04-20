import json
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from config import DATA_PATH


# -----------------------------
# DATA LOADING
# -----------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


# -----------------------------
# MODEL
# -----------------------------
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# MAIN
# -----------------------------


def run_experiment():


    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X_train.shape[1])

    model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=1)

    # predictions
    probs = model.predict(X_test).reshape(-1)
    preds = (probs > 0.5).astype(int)

    # metrics
    results = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs)
    }

    print("\n🧠 ANN (NO GA) Results")
    for k, v in results.items():
        print(f"{k.capitalize():10}: {v:.4f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "neural_network_no_ga.joblib")
    joblib.dump(model, model_path)

    print(f"\n💾 Model saved to: {model_path}")

    # -----------------------------
    # ORCHESTRATION OUTPUT
    # -----------------------------
    print("METRICS_START", json.dumps(results), "METRICS_END")
    return results



if __name__ == "__main__":
    print(run_experiment())
