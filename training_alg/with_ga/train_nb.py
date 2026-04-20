import pandas as pd
import json
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


class NaiveBayesTrainer:
    def __init__(self):
        self.model = GaussianNB()

    def load_data(self, path):
        df = pd.read_csv(path)

        X = df.drop(columns=["target"])
        y = df["target"]

        return X, y

    def load_features(self, path="selected_features.json"):
        with open(path, "r") as f:
            return json.load(f)

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

        acc = accuracy_score(y_test, preds)

        print("\n📊 Naive Bayes Results")
        print("Accuracy:", acc)
        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        # -----------------------------
        # 💾 SAVE MODEL
        # -----------------------------
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "naive_bayes.joblib")
        joblib.dump(self.model, model_path)

        print(f"\n💾 Model saved to {model_path}")

        return acc


if __name__ == "__main__":

    trainer = NaiveBayesTrainer()

    X, y = trainer.load_data("../data/processed/train.csv")

    features = trainer.load_features()

    trainer.train(X, y, features)