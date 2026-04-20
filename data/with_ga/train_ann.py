import pandas as pd
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow import keras


class ANNTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        self.model = None
        self.scaler = StandardScaler()

    def load_data(self, path):
        df = pd.read_csv(path)

        X = df.drop(columns=["target"])
        y = df["target"]

        return X, y

    def load_features(self, path="selected_features.json"):
        with open(path, "r") as f:
            return json.load(f)

    def build_model(self, input_dim):

        model = keras.Sequential([
            keras.layers.Dense(32, activation="relu", input_shape=(input_dim,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")]
        )

        return model

    def train(self, X, y, features):

        # apply GA-selected features
        X = X[features]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y
        )

        # IMPORTANT: scale for neural networks
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print(f"🧠 Training ANN on {X_train.shape[1]} features...")

        self.model = self.build_model(X_train.shape[1])

        history = self.model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=256,
            validation_split=0.2,
            verbose=1
        )

        probs = self.model.predict(X_test)
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_test, preds)

        print("\n📊 ANN Results")
        print("Accuracy:", acc)
        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        return acc

    def save_model(self, path="ann_model.keras"):
        self.model.save(path)
        print(f"\n💾 Saved ANN model to {path}")


if __name__ == "__main__":

    trainer = ANNTrainer()

    X, y = trainer.load_data("train_processed.csv")

    features = trainer.load_features()

    trainer.train(X, y, features)

    trainer.save_model()