import os
import json
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split

from training_alg.data.loader import load_dataset
from training_alg.preprocessing.preprocess import preprocess
from training_alg.with_ga.paper_ga import PaperGA
from analysis.run_roc_figures import run_figures
from analysis.ga_convergence import plot_convergence
from analysis.paper_tables import evaluate_models
from config import (
    GA_POP_SIZE,
    GA_GENERATIONS,
    SAMPLE_SIZE,
    SEED,
    PAPER_MODE,
    ROC_DIR,
    CONVERGENCE_DIR
)

# --------------------------
# GLOBAL REPRODUCIBILITY
# --------------------------
np.random.seed(SEED)
random.seed(SEED)


# --------------------------
# OUTPUT STRUCTURE
# --------------------------
OUTPUT_DIR = "outputs"
TABLE_DIR = f"{OUTPUT_DIR}/tables"
FIG_DIR = f"{OUTPUT_DIR}/figures"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(f"{FIG_DIR}/roc", exist_ok=True)
os.makedirs(f"{FIG_DIR}/ga", exist_ok=True)


# --------------------------
# MAIN PIPELINE
# --------------------------
def run():

    print("\n===================================")
    print("   PAPER REPRODUCTION PIPELINE")
    print("===================================\n")

    # --------------------------
    # 1. LOAD DATA
    # --------------------------
    print("Loading dataset...")
    X, y = load_dataset("training_alg/data/raw/creditcard.csv")

    X = preprocess(X)

    if SAMPLE_SIZE:
        print(f"DEBUG MODE: sampling {SAMPLE_SIZE}")
        X = X.sample(SAMPLE_SIZE, random_state=SEED)
        y = y.loc[X.index]

    # --------------------------
    # 2. SPLIT
    # --------------------------
    print("Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    # --------------------------
    # 3. GA FEATURE SELECTION
    # --------------------------
    print("\nRunning GA feature selection...")

    ga = PaperGA(
        pop_size=GA_POP_SIZE,
        generations=GA_GENERATIONS
    )

    vectors = ga.run_return_all_vectors(X_train, y_train)

    # Save Table 1
    pd.DataFrame([
        {"Vector": k, "Features": v}
        for k, v in vectors.items()
    ]).to_csv(f"{TABLE_DIR}/table1_ga_vectors.csv", index=False)

    plot_convergence(ga.history)

    print("GA complete.")

    # --------------------------
    # 4. TABLES 2–7
    # --------------------------
    print("\nBuilding Tables 2–7...")

    all_tables = {}

    for name, cols in vectors.items():

        print(f"Evaluating {name}...")

        cols = [c for c in cols if c in X_train.columns]

        df = evaluate_models(
            X_train[cols],
            X_test[cols],
            y_train,
            y_test
        )

        path = f"{TABLE_DIR}/{name}_table.csv"
        df.to_csv(path, index=False)

        all_tables[name] = df

    # FULL FEATURE SET (Table 7 baseline)
    full_df = evaluate_models(X_train, X_test, y_train, y_test)
    full_df.to_csv(f"{TABLE_DIR}/full_table.csv", index=False)

    # RANDOM BASELINE
    random_cols = np.random.choice(X_train.columns, 20, replace=False)

    rand_df = evaluate_models(
        X_train[random_cols],
        X_test[random_cols],
        y_train,
        y_test
    )
    rand_df.to_csv(f"{TABLE_DIR}/random_table.csv", index=False)

    # --------------------------
    # 5. ROC FIGURES 4–8
    # --------------------------
    print("\nGenerating ROC figures...")

    roc_results = run_figures(
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir=f"{FIG_DIR}/roc"
    )

    # --------------------------
    # 6. SAVE RUN SUMMARY
    # --------------------------
    summary = {
        "seed": SEED,
        "ga_pop_size": GA_POP_SIZE,
        "ga_generations": GA_GENERATIONS,
        "vectors": {k: len(v) for k, v in vectors.items()},
        "roc_auc": roc_results
    }

    with open(f"{OUTPUT_DIR}/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===================================")
    print("   PIPELINE COMPLETE")
    print("===================================\n")


if __name__ == "__main__":
    run()