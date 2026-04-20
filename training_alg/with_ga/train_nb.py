import pandas as pd
import json
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from config import DATA_PATH, FEATURE_PATH


class NaiveBayesTrainer:
    def __init__(self):
        self.model = GaussianNB()

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
    # TRAIN + EVALUATE
    # -----------------------------
    def train(self, X, y, features):

        # apply GA-selected features
        X = X[features]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print(f"📊 Training Naive Bayes on {X.shape[1]} features...")

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        results = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "auc": roc_auc_score(y_test, probs)
        }

        print("\n📊 Naive Bayes Results")
        for k, v in results.items():
            print(f"{k.capitalize():10}: {v:.4f}")

        return results

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    def save_model(self, path="models/naive_bayes.joblib"):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, path)
        print(f"\n💾 Model saved to {path}")


# -----------------------------
# MAIN
# -----------------------------


def run_experiment():


    trainer = NaiveBayesTrainer()

    X, y = trainer.load_data()
    features = trainer.load_features()

    results = trainer.train(X, y, features)

    trainer.save_model()

    print("METRICS_START", json.dumps(results), "METRICS_END")
    return results



if __name__ == "__main__":
    print(run_experiment())
