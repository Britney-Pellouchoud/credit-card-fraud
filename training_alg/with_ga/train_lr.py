import pandas as pd
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class LogisticRegressionTrainer:
    def __init__(self, random_state=42, max_iter=1000):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver="lbfgs",          # stable default
            class_weight="balanced"  # important for fraud imbalance
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

        print(f"📈 Training Logistic Regression on {X.shape[1]} features...")

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print("\n📊 Logistic Regression Results")
        print("Accuracy:", acc)
        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        return acc

    def save_model(self, path="logistic_regression.joblib"):
        joblib.dump(self.model, path)
        print(f"\n💾 Saved model to {path}")


if __name__ == "__main__":

    trainer = LogisticRegressionTrainer(
        max_iter=1000
    )

    X, y = trainer.load_data("../data/processed/train.csv")

    features = trainer.load_features()

    trainer.train(X, y, features)

    trainer.save_model()