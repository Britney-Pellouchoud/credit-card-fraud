import pandas as pd
import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class RandomForestTrainer:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,   # 🚀 speed boost
            class_weight="balanced"  # important for fraud data
        )

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

        print(f"🌲 Training Random Forest on {X.shape[1]} features...")

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print("\n📊 Random Forest Results")
        print("Accuracy:", acc)
        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        return acc

    def save_model(self, path="random_forest.joblib"):
        joblib.dump(self.model, path)
        print(f"\n💾 Saved model to {path}")


if __name__ == "__main__":

    trainer = RandomForestTrainer(
        n_estimators=100,
        max_depth=None
    )

    X, y = trainer.load_data("train_processed.csv")

    features = trainer.load_features()

    trainer.train(X, y, features)

    trainer.save_model()