import pandas as pd
import json
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class DecisionTreeTrainer:
    def __init__(self, max_depth=10, random_state=42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
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

        X = X[features]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print("\n🌳 Decision Tree Results")
        print("Accuracy:", acc)
        print(classification_report(y_test, preds))

        return acc

    def save_model(self, path="decision_tree.joblib"):
        joblib.dump(self.model, path)
        print(f"\n💾 Saved model to {path}")


if __name__ == "__main__":

    trainer = DecisionTreeTrainer(max_depth=10)

    X, y = trainer.load_data("train_processed.csv")

    features = trainer.load_features()

    trainer.train(X, y, features)

    trainer.save_model()