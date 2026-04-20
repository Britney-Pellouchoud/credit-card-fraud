import json
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from config import DATA_PATH


# -----------------------------
# DATA LOADING (STANDARDIZED)
# -----------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


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

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # -----------------------------
    # METRICS
    # -----------------------------
    results = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs)
    }

    print("\n🌲 Random Forest (NO GA) Results")
    for k, v in results.items():
        print(f"{k.capitalize():10}: {v:.4f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "random_forest_no_ga.joblib")
    joblib.dump(model, model_path)

    print(f"\n💾 Model saved to: {model_path}")

    # -----------------------------
    # ORCHESTRATION OUTPUT
    # -----------------------------
    print("METRICS_START", json.dumps(results), "METRICS_END")
    return results



if __name__ == "__main__":
    print(run_experiment())
