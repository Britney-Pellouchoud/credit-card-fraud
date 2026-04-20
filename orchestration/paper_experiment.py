import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from training_alg.data.loader import load_data_smote
from training_alg.with_ga.paper_ga import PaperGA
from training_alg.models.paper_models import get_models
from training_alg.evaluation.evaluator import evaluate_all


# -----------------------------
# PROJECT PATH
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "training_alg" / "data" / "creditcard.csv"


# -----------------------------
# LOAD DATASET
# -----------------------------
def load_dataset():
    print("\n🚀 LOADING DATASET")
    print(f"📂 Loading dataset from: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    X, y = load_data_smote(str(DATA_PATH))
    return X, y


# -----------------------------
# GA PIPELINE (FAST VERSION)
# -----------------------------
def run_ga(X, y):

    print("\n🧬 Running GA feature selection (FAST MODE)...")

    # 🔥 SMALLER SAMPLE FOR SPEED (still paper-valid idea)
    X_sample, _, y_sample, _ = train_test_split(
        X, y,
        test_size=0.7,
        stratify=y,
        random_state=42
    )

    # PAPER CORRECT SPLIT (on sample)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample,
        test_size=0.2,
        stratify=y_sample,
        random_state=42
    )

    # SMOTE ONLY ON TRAIN (correct)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # ⚡ SMALL GA (THIS IS THE SPEED FIX)
    ga = PaperGA(
        pop_size=10,        # was 20
        generations=5,      # was 10
        mutation_rate=0.02,
        random_state=42
    )

    vectors = ga.run_multi_vectors(X_train, y_train, n_vectors=5)

    return vectors, (X_train, X_test, y_train, y_test)


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_vectors(vectors, data):

    X_train, X_test, y_train, y_test = data

    models = get_models()
    results = []

    for i, cols in enumerate(vectors):

        print(f"\n📊 Evaluating Vector v{i+1} | features={len(cols)}")

        Xtr = X_train[cols]
        Xte = X_test[cols]

        for name, model in models.items():

            metrics = evaluate_all(
                model,
                Xtr, Xte,
                y_train, y_test
            )

            results.append({
                "vector": f"v{i+1}",
                "model": name,
                **metrics
            })

    return pd.DataFrame(results)


# -----------------------------
# BASELINE (NO GA)
# -----------------------------
def run_baseline(X, y):

    print("\n🚀 Running baseline (NO GA)...")

    # same sampling strategy for fairness
    X_sample, _, y_sample, _ = train_test_split(
        X, y,
        test_size=0.7,
        stratify=y,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample,
        test_size=0.2,
        stratify=y_sample,
        random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    models = get_models()
    results = []

    for name, model in models.items():

        metrics = evaluate_all(
            model,
            X_train, X_test,
            y_train, y_test
        )

        results.append({
            "vector": "FULL",
            "model": name,
            **metrics
        })

    return pd.DataFrame(results)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    X, y = load_dataset()

    vectors, data = run_ga(X, y)

    ga_results = evaluate_vectors(vectors, data)
    baseline_results = run_baseline(X, y)

    final = pd.concat([ga_results, baseline_results], ignore_index=True)

    output_path = PROJECT_ROOT / "paper_results.csv"
    final.to_csv(output_path, index=False)

    print(f"\n✅ DONE — saved results to {output_path}")