import pandas as pd
import json
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import tensorflow as tf
from tensorflow import keras

from config import DATA_PATH, FEATURE_PATH


class ANNTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        self.model = None
        self.scaler = StandardScaler()

    # -----------------------------
    # DATA LOADING (STANDARDIZED)
    # -----------------------------
    def load_data(self):
        df = pd.read_csv(DATA_PATH)

        X = df.drop(columns=["target"])
        y = df["target"]

        return X, y

    # -----------------------------
    # FEATURE LOADING (GA)
    # -----------------------------
    def load_features(self):
        with open(FEATURE_PATH, "r") as f:
            return json.load(f)

    # -----------------------------
    # MODEL
    # -----------------------------
    def build_model(self, input_dim):
        model = keras.Sequential(
            [
                keras.layers.Dense(32, activation="relu", input_shape=(input_dim,)),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(8, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )

        return model

    # -----------------------------
    # TRAIN + EVAL
    # -----------------------------
    def train(self, X, y, features):

        # apply GA features
        X = X[features]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print(f"🧠 Training ANN on {X_train.shape[1]} features...")

        self.model = self.build_model(X_train.shape[1])

        self.model.fit(
            X_train, y_train, epochs=10, batch_size=256, validation_split=0.2, verbose=1
        )

        # predictions
        probs = self.model.predict(X_test).reshape(-1)
        preds = (probs > 0.5).astype(int)

        # metrics
        results = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "auc": roc_auc_score(y_test, probs),
        }

        print("\n📊 ANN Results")
        for k, v in results.items():
            print(f"{k.capitalize():10}: {v:.4f}")

        return results

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    def save_model(self, path="models/ann_model.keras"):
        os.makedirs("models", exist_ok=True)
        self.model.save(path)
        print(f"\n💾 Saved ANN model to {path}")


# -----------------------------
# MAIN
# -----------------------------


def run_experiment():
    trainer = ANNTrainer()

    X, y = trainer.load_data()
    features = trainer.load_features()

    results = trainer.train(X, y, features)

    trainer.save_model()

    print("METRICS_START", json.dumps(results), "METRICS_END")
    return results


if __name__ == "__main__":
    print(run_experiment())
